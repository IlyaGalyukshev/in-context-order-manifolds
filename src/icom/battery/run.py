"""Battery runner: experiment YAML in, one parquet of BatteryRow out.

Idempotent: rows are keyed by (stimulus_id, question hash, model); on restart,
already-present keys are skipped.
"""

from __future__ import annotations


def run(config_path: str) -> None:
    raise NotImplementedError
