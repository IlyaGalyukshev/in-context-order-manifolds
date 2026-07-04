"""Scoring: parsers per question family + graded metrics.

- Reconstruction → exact match (secondary; floors at N≳20) and Kendall tau /
  Spearman rho over the parsed list (primary).
- Pairwise → accuracy per rank-distance bin; logit margin P(yes)-P(no) as the
  graded, format-robust score.
- Adjacency / rank → accuracy split by endpoint vs interior.
- Span → set overlap + within-span order.
- Parse failures → BatteryRow.parse_failed=True, correct=None. A high parse
  failure rate is a finding (and a steering outcome measure), never silently
  folded into errors.
"""

from __future__ import annotations

from icom.generator.schemas import BatteryRow, Question


def score_completion(question: Question, raw: dict, model: str, git_sha: str, seed: int) -> BatteryRow:
    raise NotImplementedError
