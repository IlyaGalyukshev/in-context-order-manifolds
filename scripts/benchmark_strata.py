#!/usr/bin/env python
"""Benchmark strata — full, analysis-ready breakdown of a generated BCS dataset.

Joins questions.jsonl to stimuli.jsonl (by content_key) and reports, per query
CATEGORY (family), the counts and structure needed to plan and later interpret
the behavioral analysis:
  - totals, interior-answerable share, endpoint-diagnostic share
  - answer-key spread (distinct keys, top-key share) -> a degeneracy guard
  - chance baseline and rank-distance distribution
  - coverage matrices: family x N (interior-answerable), structure x family,
    family x difficulty.

Writes a machine-readable JSON and a human Markdown report.

  python scripts/benchmark_strata.py --data data/bcs \
      --md docs/benchmark_strata.md --json data/bcs/strata.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# forced-choice families -> #options (for chance); others annotated below.
CHANCE = {
    "reconstruction": "Kendall tau = 0",
    "pairwise": "0.50 (2-choice)",
    "rank": "~1/N",
    "order_query": "0.50 (2-choice + 'undetermined')",
    "betweenness": "0.33 (3-choice)",
    "successor": "~1/N",
    "predecessor": "~1/N",
    "count_between": "const-0 ~26%; report by rank_distance / MAE, not 1/(N-1)",
    "comparative_distance": "0.50 (2-choice)",
    "extremes": "~1/N (endpoint-diagnostic)",
}
# families whose keys are meant to concentrate (numbers / lists) -> skip the
# top-key degeneracy flag for them.
NUMERIC_OR_LIST = {"count_between", "rank", "reconstruction"}


def load_jsonl(p: Path):
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def interior_flag(q):
    return bool(q.get("interior_ok") or q.get("both_interior"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dataset dir with stimuli/questions jsonl")
    ap.add_argument("--md", default=None, help="Markdown report out path")
    ap.add_argument("--json", default=None, help="JSON strata out path")
    args = ap.parse_args()
    data = Path(args.data)

    stimuli = load_jsonl(data / "stimuli.jsonl")
    questions = load_jsonl(data / "questions.jsonl")
    nulls = load_jsonl(data / "stimuli_null.jsonl")
    meta = json.loads((data / "meta.json").read_text()) if (data / "meta.json").exists() else {}

    # content_key -> stimulus attrs (dedupe conditions; keep the set of conditions)
    attrs = {}
    conds_of = defaultdict(set)
    struct_ct = Counter()
    cond_ct = Counter()
    for s in stimuli:
        ck = s["content_key"]
        structure = s.get("structure", "total_order")
        conds_of[ck].add(s.get("condition", "?"))
        cond_ct[(structure, s.get("condition", "?"))] += 1
        struct_ct[structure] += 1
        if ck not in attrs:
            attrs[ck] = {"family": s.get("family", "?"), "N": s.get("n_items"),
                         "difficulty": s.get("difficulty", "na"), "structure": structure}

    # ---- per-family aggregation
    fam = defaultdict(lambda: {"n": 0, "interior": 0, "endpoint": 0,
                               "keys": Counter(), "rank_d": [], "N": Counter()})
    fam_by_N = defaultdict(Counter)        # family -> N -> interior-answerable count
    struct_fam = defaultdict(Counter)      # structure -> family -> count
    fam_by_diff = defaultdict(Counter)     # family -> difficulty -> interior count
    orphan = 0
    for q in questions:
        a = attrs.get(q["stimulus_content_key"])
        if a is None:
            orphan += 1
            continue
        f = q["family"]
        rec = fam[f]
        rec["n"] += 1
        rec["N"][a["N"]] += 1
        is_int = interior_flag(q)
        rec["interior"] += 1 if is_int else 0
        rec["endpoint"] += 1 if q.get("is_endpoint") else 0
        rec["keys"][str(q.get("answer_key"))] += 1
        if "rank_distance" in q:
            rec["rank_d"].append(int(q["rank_distance"]))
        struct_fam[a["structure"]][f] += 1
        if is_int:
            fam_by_N[f][a["N"]] += 1
            fam_by_diff[f][a["difficulty"]] += 1

    # ---- assemble strata
    families = {}
    for f, rec in sorted(fam.items()):
        n = rec["n"]
        top_key, top_ct = rec["keys"].most_common(1)[0] if rec["keys"] else ("", 0)
        rd = rec["rank_d"]
        families[f] = {
            "n": n,
            "interior_answerable": rec["interior"],
            "interior_pct": round(100 * rec["interior"] / n, 1) if n else 0.0,
            "endpoint_diagnostic": rec["endpoint"],
            "distinct_keys": len(rec["keys"]),
            "top_key_share": round(top_ct / n, 3) if n else 0.0,
            "chance": CHANCE.get(f, "?"),
            "rank_distance_hist": dict(sorted(Counter(rd).items())) if rd else {},
            "by_N": dict(sorted(rec["N"].items())),
        }
    # degeneracy guard: forced-choice / name families whose single top key is huge
    flags = []
    for f, d in families.items():
        if f in NUMERIC_OR_LIST:
            continue
        thresh = 0.6 if f == "extremes" else 0.55   # extremes has only 2 keys/stim
        if d["top_key_share"] > thresh and d["distinct_keys"] > 2:
            flags.append(f"{f}: top-key share {d['top_key_share']} (>{thresh})")

    strata = {
        "dataset": str(data),
        "meta": meta,
        "n_stimuli": len(stimuli),
        "n_stimuli_by_structure": dict(struct_ct),
        "n_stimuli_by_structure_condition": {f"{k[0]}/{k[1]}": v for k, v in sorted(cond_ct.items())},
        "n_null_twins": len(nulls),
        "n_questions": len(questions),
        "orphan_questions": orphan,
        "families": families,
        "coverage_family_by_N_interior": {f: dict(sorted(c.items())) for f, c in fam_by_N.items()},
        "structure_by_family": {s: dict(sorted(c.items())) for s, c in struct_fam.items()},
        "family_by_difficulty_interior": {f: dict(c) for f, c in fam_by_diff.items()},
        "degeneracy_flags": flags,
    }

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(strata, indent=2))
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(render_md(strata))
    print(f"stimuli={len(stimuli)} questions={len(questions)} families={len(families)} "
          f"orphans={orphan} flags={len(flags)}")
    for f in flags:
        print("  DEGENERACY FLAG:", f)


def render_md(s: dict) -> str:
    L = ["# BCS benchmark strata",
         "",
         f"*Generated from `{s['dataset']}` — analysis-ready breakdown of every "
         "query category. Interior-answerable = all involved entities in ranks "
         "3..N−2 (the confound-clean subset used for geometry-linked behavior). "
         "Endpoint-diagnostic = leaks role/frequency, reported as a control only.*",
         ""]
    m = s.get("meta", {})
    if m:
        L += [f"- families: `{', '.join(m.get('families', []))}`  ·  N-grid: "
              f"`{m.get('n_grid')}`  ·  per-cell: {m.get('per_cell')}  ·  "
              f"difficulty: {m.get('difficulty')}  ·  degree: {m.get('degree')}",
              f"- gate_failures at generation: **{m.get('gate_failures', '?')}**", ""]
    L += [f"- **stimuli:** {s['n_stimuli']}  {s['n_stimuli_by_structure']}",
          f"- **coherence-null twins:** {s['n_null_twins']}",
          f"- **questions:** {s['n_questions']}  (orphans: {s['orphan_questions']})",
          f"- **degeneracy flags:** {s['degeneracy_flags'] or 'none'}",
          ""]

    L += ["## Per-category strata", "",
          "| family | n | interior | interior % | endpoint | #keys | top-key share | chance |",
          "|---|--:|--:|--:|--:|--:|--:|---|"]
    for f, d in s["families"].items():
        L.append(f"| `{f}` | {d['n']} | {d['interior_answerable']} | {d['interior_pct']} | "
                 f"{d['endpoint_diagnostic']} | {d['distinct_keys']} | {d['top_key_share']} | "
                 f"{d['chance']} |")
    L.append("")

    L += ["## Interior-answerable coverage (family × N)", "",
          "*How many confound-clean, interior-only questions exist per length — "
          "the power budget for interior geometry-linked behavior.*", ""]
    Ns = sorted({n for c in s["coverage_family_by_N_interior"].values() for n in c})
    L.append("| family | " + " | ".join(f"N={n}" for n in Ns) + " |")
    L.append("|---|" + "|".join("--:" for _ in Ns) + "|")
    for f, c in sorted(s["coverage_family_by_N_interior"].items()):
        L.append(f"| `{f}` | " + " | ".join(str(c.get(n, 0)) for n in Ns) + " |")
    L.append("")

    L += ["## Structure × family (question counts)", ""]
    for struct, c in s["structure_by_family"].items():
        L.append(f"- **{struct}**: " + ", ".join(f"`{f}`={n}" for f, n in sorted(c.items())))
    L.append("")

    L += ["## rank-distance histograms (integration reach)", "",
          "*Distance-stratified categories: d=1 is directly stated by a card; "
          "d≥2 requires transitive integration (the real difficulty axis).*", ""]
    for f, d in s["families"].items():
        if d["rank_distance_hist"]:
            L.append(f"- `{f}`: {d['rank_distance_hist']}")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    main()
