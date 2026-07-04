"""Deterministic seeding. Every artifact is reproducible from (config, seed);
derived seeds are spawned per stimulus via numpy SeedSequence, never global state."""

from __future__ import annotations


def seed_everything(seed: int) -> None:
    raise NotImplementedError


def child_seed(root_seed: int, *keys) -> int:
    raise NotImplementedError
