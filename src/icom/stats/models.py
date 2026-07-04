"""Track A analysis specification.

Condition and N causally drive BOTH manifold quality and accuracy, so pooled
regressions are spurious by construction. The claim of interest is the
WITHIN-CELL slope: mixed-effects (logistic for binary outcomes) of per-question
accuracy on q_content_partial, fit within (condition × N × model) cells with
stimulus as a random intercept; covariates: attention entropy, mean item
position, question family. Report ΔAUC/ΔR² over the covariate-only baseline,
alongside the quality score's reliability (geometry/reliability.py).
"""

from __future__ import annotations


def track_a_regression(battery_df, geometry_df):
    raise NotImplementedError
