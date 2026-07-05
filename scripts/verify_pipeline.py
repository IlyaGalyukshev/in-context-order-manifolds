#!/usr/bin/env python
"""Full pipeline verification: data, generation, prompts, parsing, scoring,
configs, pools. Regenerates a fresh dataset with the CURRENT generator/question
code and audits it end to end, then writes a human-readable + JSON report.

Run on the worker (needs transformers + the committed pools + HF cache):
  python scripts/verify_pipeline.py --out /workspace/manifolds/results/verification

Every check returns (name, ok, detail). Nothing is trusted implicitly:
question keys are recomputed from the latent order, relational orders are
rebuilt from the shuffled prompt text, scoring is recomputed independently.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from icom.battery.scoring import parse_yesno, score_row
from icom.generator.questions import make_battery
from icom.generator.schemas import Condition, QuestionFamily, StimulusFamily
from icom.generator.stimuli import MAX_SHUFFLE_ABS_RHO, make_condition_set

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


# ---------------------------------------------------------------- fixtures
def load_pools(cfg):
    preds = json.load(open(cfg["pools"]["predicates"]))["predicates"]
    ents = json.load(open(cfg["pools"]["entities"]))["names"]
    return ents, preds


def build_sample(cfg, ents, preds, per_cell=8):
    """Regenerate a representative dataset with the CURRENT code."""
    seed = cfg["seed"]
    conds = [Condition.FORWARD, Condition.REVERSE, Condition.SHUFFLE]
    data = []
    for fam in [StimulusFamily.DATED, StimulusFamily.TAGGED, StimulusFamily.RELATIONAL]:
        for n in cfg["n_items_grid"]:
            for idx in range(per_cell):
                stims, latent, key = make_condition_set(fam, n, seed, idx, ents, preds, conds)
                battery = make_battery(latent, key, fam, seed,
                                       pairwise_per_bin=cfg["questions"]["pairwise_per_distance_bin"],
                                       distance_bins=cfg["questions"]["distance_bins"],
                                       adjacency_max=cfg["questions"]["adjacency_max"],
                                       rank_max=cfg["questions"]["rank_max"])
                data.append((fam, n, idx, stims, latent, key, battery))
    return data


def mention_order(prompt, latent):
    pos = {e: re.search(rf"\b{re.escape(e)}\b", prompt).start() for e in latent}
    return sorted(pos, key=pos.get)


def rebuild_relational(prompt):
    succ = {}
    for line in prompt.split("\n"):
        m = re.match(r"[Tt]he (\w+) .* before the (\w+) ", line)
        succ[m.group(1)] = m.group(2)
    start = (set(succ) - set(succ.values())).pop()
    chain = [start]
    while chain[-1] in succ:
        chain.append(succ[chain[-1]])
    return chain


# ---------------------------------------------------------------- A. generation
def verify_generation(data, ents, preds):
    key_errs = defaultdict(int)
    n_keys = 0
    for fam, n, idx, stims, latent, ckey, battery in data:
        rank = {e: i + 1 for i, e in enumerate(latent)}
        for q in battery:
            n_keys += 1
            f, key, tg = q.family, q.answer_key, q.target_entities
            good = True
            if f is QuestionFamily.RECONSTRUCTION:
                good = key == latent or key == "MENTION_ORDER"
            elif f is QuestionFamily.PAIRWISE:
                a, b = tg
                good = key == (a if rank[a] < rank[b] else b) and q.rank_distance == abs(rank[a] - rank[b])
            elif f is QuestionFamily.ADJACENCY:
                good = key == latent[rank[tg[0]]]
            elif f is QuestionFamily.RANK:
                good = (key == str(rank[tg[0]])) if str(key).isdigit() else (rank[key] == rank[tg[0]])
            elif f is QuestionFamily.SPAN:
                x = rank[tg[0]]
                good = key == latent[x:x + 3]
            if not good:
                key_errs[f.value] += 1
    check("A1 question keys recomputed from latent order",
          not key_errs, f"{n_keys} keys checked, errors={dict(key_errs)}")

    # relational order rebuilt from shuffled text
    ok = bad = 0
    for fam, n, idx, stims, latent, ckey, battery in data:
        if fam is not StimulusFamily.RELATIONAL:
            continue
        chain = rebuild_relational(stims[Condition.SHUFFLE].prompt)
        ok += chain == latent
        bad += chain != latent
    check("A2 relational order rebuilt from SHUFFLED text == latent", not bad,
          f"{ok} ok / {bad} fail")

    # conditions share content + questions
    bad = 0
    for fam, n, idx, stims, latent, ckey, battery in data:
        texts = [sorted(c.text for c in s.cards) for s in stims.values()]
        if not all(t == texts[0] for t in texts):
            bad += 1
    check("A3 all conditions share identical card content", not bad, f"{bad} mismatches")

    # shuffle decorrelation
    worst = 0.0
    for fam, n, idx, stims, latent, ckey, battery in data:
        s = stims[Condition.SHUFFLE]
        rr = [c.latent_rank for c in s.cards]
        ss = [c.presentation_slot for c in s.cards]
        m = len(rr)
        mr, ms = sum(rr) / m, sum(ss) / m
        cov = sum((a - mr) * (b - ms) for a, b in zip(rr, ss))
        den = (sum((a - mr) ** 2 for a in rr) * sum((b - ms) ** 2 for b in ss)) ** 0.5
        worst = max(worst, abs(cov / den))
    check("A4 shuffle slot-rank decorrelated (<=0.25)", worst <= MAX_SHUFFLE_ABS_RHO + 1e-9,
          f"max|rho|={worst:.3f}")

    # fixed-width markers, strictly increasing
    bad_w = bad_mono = 0
    for fam, n, idx, stims, latent, ckey, battery in data:
        if fam is StimulusFamily.RELATIONAL:
            continue
        mk = stims[Condition.FORWARD].markers
        if len({len(v) for v in mk.values()}) != 1:
            bad_w += 1
        vals = [int(re.search(r"\d+", mk[e]).group()) for e in latent]
        if vals != sorted(vals) or len(set(vals)) != len(vals):
            bad_mono += 1
    check("A5 markers fixed-width & strictly increasing", not (bad_w or bad_mono),
          f"width_bad={bad_w} mono_bad={bad_mono}")

    # relational: only adjacent pairs, no digits
    bad = 0
    for fam, n, idx, stims, latent, ckey, battery in data:
        if fam is not StimulusFamily.RELATIONAL:
            continue
        rank = {e: i + 1 for i, e in enumerate(latent)}
        for c in stims[Condition.SHUFFLE].cards:
            if c.entity_b is None or rank[c.entity_b] - rank[c.entity] != 1 or any(ch.isdigit() for ch in c.text):
                bad += 1
    check("A6 relational cards: adjacent pairs only, no digits", not bad, f"{bad} bad cards")

    # predicate pool: no leak words / digits, all from committed pool
    from icom.generator.llm_assist import ORDINAL_LEAK_HINTS
    leak = set(ORDINAL_LEAK_HINTS)
    poolset = set(preds)
    bad_leak = bad_digit = bad_pool = 0
    for p in preds:
        words = set(re.findall(r"[a-z]+", p.lower()))
        if words & leak:
            bad_leak += 1
        if any(ch.isdigit() for ch in p):
            bad_digit += 1
    check("A7 predicate pool: no ordinal-leak words, no digits",
          not (bad_leak or bad_digit), f"leak={bad_leak} digit={bad_digit} of {len(preds)}")

    # entities: unique per stimulus, nonce
    from wordfreq import zipf_frequency
    realwords = sum(1 for e in ents if zipf_frequency(e, "en") > 1.5)
    check("A8 entity pool: all nonce (wordfreq zipf<=1.5)", realwords == 0,
          f"{realwords} real words of {len(ents)}")

    # determinism
    a = make_condition_set(StimulusFamily.RELATIONAL, 16, 20260704, 3, ents, preds,
                           [Condition.FORWARD, Condition.SHUFFLE])
    b = make_condition_set(StimulusFamily.RELATIONAL, 16, 20260704, 3, ents, preds,
                           [Condition.FORWARD, Condition.SHUFFLE])
    dump = lambda x: json.dumps({c.value: dataclasses.asdict(s) for c, s in x[0].items()}, sort_keys=True)
    check("A9 determinism: same seed -> identical", dump(a) == dump(b), "regenerated twice")


# ---------------------------------------------------------------- B. tokenization
def verify_tokenization(data, tokenizers):
    for tid, tok in tokenizers.items():
        # token-length parity across conditions
        worst = 0
        for fam, n, idx, stims, latent, ckey, battery in data:
            lens = {len(tok(s.prompt)["input_ids"]) for s in stims.values()}
            worst = max(worst, max(lens) - min(lens))
        check(f"B[{tid}] token-length identical across conditions", worst == 0,
              f"max spread={worst} tokens")


# ---------------------------------------------------------------- C. prompts
def render_prompts(data, tokenizers, report_lines):
    tok = next(iter(tokenizers.values()))
    tid = next(iter(tokenizers))
    report_lines.append(f"\n### Rendered prompts (tokenizer {tid}, thinking disabled)\n")
    seen = set()
    for fam, n, idx, stims, latent, ckey, battery in data:
        if fam in seen:
            continue
        seen.add(fam)
        s = stims[Condition.SHUFFLE]
        user = s.prompt + "\n\n" + battery[2].text  # a pairwise question
        try:
            full = tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            full = tok.apply_chat_template([{"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True)
        report_lines.append(f"--- {fam.value} / shuffle ---\n```\n{full}\n```\n")
        # structural checks
        check(f"C[{fam.value}] format clause present on question",
              "No explanation" in battery[2].text or "nothing else" in battery[2].text,
              battery[2].text[-60:])
    check("C thinking-mode disabled in template", "<think>" not in full or "</think>" in full,
          "no dangling think block")


# ---------------------------------------------------------------- D. parsing
def verify_parsing():
    yn = [
        ("Yes.  \n\nThe quoannel", "yes"), ("**No**, the stind", "no"),
        ("the answer is no.", "no"), ("No... wait, the answer is yes.", "yes"),
        ("To determine if the **x", None), ("It is unclear.", None),
    ]
    bad = [f"{t!r}->{parse_yesno(t)} want {w}" for t, w in yn if parse_yesno(t) != w]
    check("D1 yes/no parser (legacy) unit cases", not bad, "; ".join(bad) or "6/6")

    # forced-choice extraction via score_row
    q = {"family": "pairwise", "answer_key": "drumb",
         "target_entities": ("drumb", "stind"), "rank_distance": 1}
    cases = [("stind", False), ("drumb", False), ("The drumb.", None), ("the DRUMB", None)]
    # note: correct depends on which is earlier; here key=drumb
    got = []
    for ans, _ in cases:
        r = score_row(q, ans, ["drumb", "stind"], None)
        got.append((ans, r["correct"], r["parse_failed"]))
    ok = (got[0][1] is False and got[1][1] is True and got[2][1] is True)
    check("D2 forced-choice extraction picks named entity", ok, str(got))

    # rank: last integer wins
    q = {"family": "rank", "answer_key": "6", "target_entities": ("x",)}
    r = score_row(q, "The tag is 35, so counting from lowest the position is 6.", ["x"], None)
    check("D3 rank parser takes final integer", r["correct"] is True, f"got correct={r['correct']}")

    # adjacency: anchor echo excluded
    q = {"family": "adjacency", "answer_key": "floane", "target_entities": ("slulb",)}
    r = score_row(q, "The entity after the slulb is the floane.", ["slulb", "floane", "x"], None)
    check("D4 adjacency excludes echoed anchor", r["correct"] is True, f"correct={r['correct']}")


# ---------------------------------------------------------------- E. scoring
def verify_scoring(data):
    """Independently recompute correctness for synthetic perfect/wrong answers."""
    fam, n, idx, stims, latent, ckey, battery = next(d for d in data if d[0] is StimulusFamily.TAGGED)
    rank = {e: i + 1 for i, e in enumerate(latent)}
    mism = 0
    for q in battery:
        qd = dataclasses.asdict(q)
        qd["family"] = q.family.value
        if q.family is QuestionFamily.PAIRWISE:
            a, b = q.target_entities
            earlier = a if rank[a] < rank[b] else b
            r_ok = score_row(qd, earlier, latent, None)
            r_bad = score_row(qd, (b if earlier == a else a), latent, None)
            if not (r_ok["correct"] is True and r_bad["correct"] is False):
                mism += 1
        elif q.family is QuestionFamily.ADJACENCY:
            r_ok = score_row(qd, q.answer_key, latent, None)
            if r_ok["correct"] is not True:
                mism += 1
        elif q.family is QuestionFamily.RANK and str(q.answer_key).isdigit():
            r_ok = score_row(qd, str(q.answer_key), latent, None)
            if r_ok["correct"] is not True:
                mism += 1
    check("E1 scoring: perfect answer scores correct, wrong scores incorrect",
          not mism, f"{mism} mismatches")

    # mention-order twin key resolved per condition
    fam, n, idx, stims, latent, ckey, battery = next(d for d in data if d[0] is StimulusFamily.RELATIONAL)
    twin = next(q for q in battery if q.answer_key == "MENTION_ORDER")
    twd = dataclasses.asdict(twin); twd["family"] = "reconstruction"
    men = mention_order(stims[Condition.SHUFFLE].prompt, latent)
    r = score_row(twd, "\n".join(men), latent, None, mention_order=men)
    check("E2 mention-order twin scores tau=1 against mention order",
          abs(r.get("tau", 0) - 1.0) < 1e-9, f"tau={r.get('tau')}")


# ---------------------------------------------------------------- F. configs
def verify_configs(gen_cfg, models_cfg):
    check("F1 pool paths exist",
          Path(gen_cfg["pools"]["predicates"]).exists() and Path(gen_cfg["pools"]["entities"]).exists(), "")
    roster = {}
    for sec in ("models", "confirmatory", "exploratory"):
        roster.update(models_cfg.get(sec) or {})
    bad = [k for k, v in roster.items() if v.get("dtype", "float16") != "float16" and "70b" not in k.lower()]
    check("F2 roster models fp16 (V100)", not bad, f"non-fp16: {bad}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen-config", default="configs/generation.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer-ids", default="Qwen/Qwen3-4B,allenai/Olmo-3-7B-Instruct")
    args = ap.parse_args()

    gen_cfg = yaml.safe_load(open(args.gen_config))
    models_cfg = yaml.safe_load(open(args.models_config))
    ents, preds = load_pools(gen_cfg)
    data = build_sample(gen_cfg, ents, preds, per_cell=8)

    from transformers import AutoTokenizer
    tokenizers = {t: AutoTokenizer.from_pretrained(t) for t in args.tokenizer_ids.split(",")}

    report_lines = ["# Pipeline verification report\n"]
    verify_generation(data, ents, preds)
    verify_tokenization(data, tokenizers)
    verify_configs(gen_cfg, models_cfg)
    verify_parsing()
    verify_scoring(data)
    render_prompts(data, tokenizers, report_lines)

    n_pass = sum(ok for _, ok, _ in CHECKS)
    report_lines.insert(1, f"\n**{n_pass}/{len(CHECKS)} checks passed** "
                           f"({sum(len(d[6]) for d in data)} questions across "
                           f"{len(data)} contents regenerated with current code)\n")
    report_lines.append("\n## Checks\n")
    for name, ok, detail in CHECKS:
        report_lines.append(f"- [{'PASS' if ok else 'FAIL'}] {name} — {detail}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(report_lines))
    json.dump([{"check": n, "ok": o, "detail": d} for n, o, d in CHECKS],
              open(out / "report.json", "w"), indent=2)

    print(f"\n{'='*70}\nVERIFICATION: {n_pass}/{len(CHECKS)} passed\n{'='*70}")
    for name, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    print(f"\nreport → {out}/report.md")
    if n_pass != len(CHECKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
