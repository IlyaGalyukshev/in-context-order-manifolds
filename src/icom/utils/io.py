"""Parquet/npz IO with provenance. Every table row carries git_sha, model,
layer, pooling, family, condition, N, seed — enforced at write time."""

from __future__ import annotations


def write_rows(rows, path: str) -> None:
    raise NotImplementedError


def current_git_sha() -> str:
    raise NotImplementedError
