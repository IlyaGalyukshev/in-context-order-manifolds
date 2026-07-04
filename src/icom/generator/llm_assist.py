"""LLM-assisted authoring of generator pools (OpenRouter).

Strict boundary, so benchmark reproducibility never depends on a remote model:
LLMs are used OFFLINE, at pool-authoring time only —
  1. author_event_predicates: propose a large pool of neutral event predicates
     ("registered a wobbling", "absorbed a festrick") that (a) carry no
     temporal/ordinal connotation, (b) are semantically interchangeable,
     (c) hold token count within a fixed band per roster tokenizer;
  2. review_pool: a second model adversarially screens the pool for ordinal
     leakage (predicates implying begin/end/growth), cultural/real-entity
     echoes, and tokenizer irregularities;
  3. the surviving pool is written to data/pools/*.json and COMMITTED.

The deterministic generator (stimuli.py) then only samples from committed
pools using (config, seed). Reruns never touch the API; the API key's absence
must never break `generate_dataset.py`.

Credentials: OPENROUTER_API_KEY / OPENROUTER_BASE_URL / OPENROUTER_PROXY from
the environment or a local .env (see .env.example). On the DGX workspace the
filled .env lives at /workspace/manifolds/.env.
"""

from __future__ import annotations


def author_event_predicates(n: int, model: str, seed_prompt_version: str) -> list[str]:
    raise NotImplementedError


def review_pool(candidates: list[str], model: str) -> list[str]:
    """Return the subset passing the ordinal-leakage / novelty screen."""
    raise NotImplementedError
