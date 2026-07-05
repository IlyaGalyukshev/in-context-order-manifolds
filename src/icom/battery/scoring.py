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
              logit_margin: float | None) -> dict:
    fam = question["family"]
    key = question["answer_key"]
    text = completion.strip()
    r: dict = {"parse_failed": False, "correct": None, "score": np.nan,
               "tau": np.nan, "coverage": np.nan, "exact_match": None}

    if fam == "pairwise":
        m = re.search(r"\b(yes|no)\b", text.lower())
        if m is None and logit_margin is None:
            r["parse_failed"] = True
            return r
        pred = m.group(1) if m else ("yes" if logit_margin > 0 else "no")
        r["correct"] = pred == key
        r["score"] = float(r["correct"])

    elif fam in ("adjacency",) or (fam == "rank" and not str(key).isdigit()):
        ents = _extract_entities(text, vocab)
        if not ents:
            r["parse_failed"] = True
            return r
        r["correct"] = ents[0] == key
        r["score"] = float(r["correct"])

    elif fam == "rank":  # numeric subtype
        m = re.search(r"\b(\d{1,3})\b", text)
        if m is None:
            r["parse_failed"] = True
            return r
        r["correct"] = int(m.group(1)) == int(key)
        r["score"] = float(r["correct"])

    elif fam in ("reconstruction", "span"):
        pred = _extract_entities(text, vocab)
        gold = list(key)
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

    return r
