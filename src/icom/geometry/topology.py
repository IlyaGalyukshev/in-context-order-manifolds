"""Topology and intrinsic dimension — gated to N ≥ 24.

Persistent homology (arc vs loop vs fragments) and TwoNN intrinsic dimension
are statistically meaningless on fewer points; callers passing n_items < 24
get None fields, by contract with GeometryRow.
"""

from __future__ import annotations

MIN_ITEMS_FOR_TOPOLOGY = 24


def topology_summary(points) -> dict | None:
    raise NotImplementedError
