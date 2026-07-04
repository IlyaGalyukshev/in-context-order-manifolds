"""Extraction runner: experiment YAML in, pooled .npz per stimulus out.

Idempotent per stimulus (keyed by stimulus_id + model + pooling).
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError
