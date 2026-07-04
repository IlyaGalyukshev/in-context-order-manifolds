"""Deterministic seeding. Every artifact is reproducible from (config, seed);
derived seeds are spawned via SeedSequence from (root_seed, *keys), never
global state. Keys are stringified so ("tagged", 16, 3) is stable."""

from __future__ import annotations

import hashlib

import numpy as np


def child_seed(root_seed: int, *keys) -> int:
    payload = f"{root_seed}|" + "|".join(map(str, keys))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def rng_for(root_seed: int, *keys) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(child_seed(root_seed, *keys)))
