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
    m = re.match(r"^(yes|no)\b", t)
    if m:
        return m.group(1)
    concl = re.findall(r"(?:answer(?:\s+is)?[:,]?|therefore[,:]?|thus[,:]?|so[,:])\s+(yes|no)\b", t)
    if concl:
        return concl[-1]
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
              logit_margin: float | None) -> dict:
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
        pred = parse_yesno(text)
        if pred is None:
            r["parse_failed"] = True
            return r
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
