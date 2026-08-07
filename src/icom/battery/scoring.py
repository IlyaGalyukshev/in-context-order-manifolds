"""Scoring: deterministic parsers per question family + graded metrics.

Philosophy: parsers extract structure (entity mentions, yes/no, integers) from
possibly-verbose completions rather than demanding exact strings. Parse
failures are a separate reported category (correct=None), never silently
scored wrong. Every row keeps the raw completion for later eyeballing.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.stats import kendalltau


def parse_yesno(text: str) -> str | None:
    """Deterministic yes/no verdict extraction, in priority order:
    1. leading yes/no (after markdown noise) — the direct answer;
    2. an explicit conclusion ("the answer is X", "therefore X") — LAST wins,
       because narrating models may consider both before concluding;
    3. last standalone yes/no anywhere;
    4. None (parse failure — e.g. narration truncated before any verdict).
    First-occurrence parsing is WRONG for narrating models ("...not no, but
    rather yes") — validated against an eyeballed sample in the pilot."""
    t = re.sub(r"[*_#>`]+", " ", text.strip().lower())
    t = re.sub(r"\s+", " ", t).strip()
    # strong conclusion ("the answer is X") overrides everything incl. a
    # leading verdict — it captures explicit self-correction; last one wins
    strong = re.findall(r"answer(?:\s+is)?[:,]?\s+(yes|no)\b", t)
    if strong:
        return strong[-1]
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    weak = re.findall(r"(?:therefore|thus|so)[,:]?\s+(yes|no)\b", t)
    if weak:
        return weak[-1]
    words = re.findall(r"\b(yes|no)\b", t)
    return words[-1] if words else None


def _extract_entities(text: str, vocab: list[str]) -> list[str]:
    """Entities of this stimulus in order of first appearance, no repeats."""
    text = text.lower()
    hits = []
    for e in vocab:
        for m in re.finditer(rf"\b{re.escape(e)}\b", text):
            hits.append((m.start(), e))
            break
    return [e for _, e in sorted(hits)]


def score_row(question: dict, completion: str, vocab: list[str],
              logit_margin: float | None, mention_order: list[str] | None = None) -> dict:
    fam = question["family"]
    key = question["answer_key"]
    text = completion.strip()
    r: dict = {"parse_failed": False, "correct": None, "score": np.nan,
               "tau": np.nan, "coverage": np.nan, "exact_match": None}

    # For anchored families the question names an anchor entity X that is
    # never the answer; models echo it ("...above the glump is the...") and
    # include it in span lists — exclude it from extraction.
    if fam in ("adjacency", "span"):
        anchors = set(question.get("target_entities") or ())
        vocab = [e for e in vocab if e not in anchors]

    if fam == "pairwise":
        # forced choice: the answer is one of the two named candidates
        cands = list(question.get("target_entities") or ())
        ents = _extract_entities(text, cands)
        if not ents:
            pred = parse_yesno(text)  # legacy yes/no data
            if pred is None:
                r["parse_failed"] = True
                return r
            r["correct"] = pred == key
        else:
            r["correct"] = ents[0] == key
        r["score"] = float(r["correct"])

    elif fam in ("adjacency",) or (fam == "rank" and not str(key).isdigit()):
        ents = _extract_entities(text, vocab)
        if not ents:
            r["parse_failed"] = True
            return r
        r["correct"] = ents[0] == key
        r["score"] = float(r["correct"])

    elif fam == "rank":  # numeric subtype
        # LAST integer: models narrate ("floane is Tag 35 ... so position 6")
        # and conclude at the end; the first int is usually the tag value.
        ints = re.findall(r"\b(\d{1,3})\b", text)
        if not ints:
            r["parse_failed"] = True
            return r
        r["correct"] = int(ints[-1]) == int(key)
        r["score"] = float(r["correct"])

    elif fam in ("reconstruction", "span"):
        pred = _extract_entities(text, vocab)
        # the mention-order control twin's key depends on the CONDITION
        # (mention order differs per presentation), so it is computed at
        # scoring time from the stimulus and passed in by the runner
        gold = mention_order if str(key) == "MENTION_ORDER" else list(key)
        if not pred:
            r["parse_failed"] = True
            return r
        r["exact_match"] = pred == gold
        common = [e for e in pred if e in gold]
        r["coverage"] = len(set(common)) / len(gold)
        if len(common) >= 2:
            gold_pos = {e: i for i, e in enumerate(gold)}
            tau, _ = kendalltau(range(len(common)), [gold_pos[e] for e in common])
            r["tau"] = float(tau)
        if fam == "span":
            r["score"] = len(set(pred[:3]) & set(gold)) / 3.0
            r["correct"] = pred[:3] == gold
        else:
            r["score"] = r["tau"] if not np.isnan(r["tau"]) else 0.0
            r["correct"] = r["exact_match"]

    # --- metric + cyclic families (v2.1/v2.2): entity-name or integer answers ---
    elif fam in ("betweenness", "extremes", "successor", "predecessor",
                 "comparative_distance", "cyclic_successor", "cyclic_predecessor",
                 "cyclic_order"):
        te = list(question.get("target_entities") or ())
        # exclude the ANCHOR/PIVOT entity (te[0]) that the question names and the
        # model echoes — it is never the answer for these families.
        anchored = fam in ("successor", "predecessor", "comparative_distance",
                           "cyclic_successor", "cyclic_predecessor", "cyclic_order")
        local = [e for e in vocab if not (anchored and te and e == te[0])]
        ents = _extract_entities(text, local)
        if not ents:
            # model that only echoed the named cue/anchor = a WRONG answer, not a
            # parse failure; a genuinely entity-free completion is the real pf.
            if anchored and te and _extract_entities(text, [te[0]]):
                r["correct"] = False
                r["score"] = 0.0
            else:
                r["parse_failed"] = True
            return r
        r["correct"] = ents[0] == key
        r["score"] = float(r["correct"])

    elif fam in ("count_between", "cyclic_distance"):  # integer answer
        ints = re.findall(r"\b(\d{1,3})\b", text)      # LAST int (models narrate first)
        if not ints:
            r["parse_failed"] = True
            return r
        r["correct"] = int(ints[-1]) == int(key)
        r["score"] = float(r["correct"])

    elif fam == "order_query":  # a named candidate, else 'undetermined'
        ents = _extract_entities(text, list(question.get("target_entities") or ()))
        if ents:
            pred = ents[0]
        elif re.search(r"\bundetermined\b", text, re.I):
            pred = "undetermined"
        else:
            r["parse_failed"] = True
            return r
        r["correct"] = pred == key
        r["score"] = float(r["correct"])

    return r
