"""LLM-assisted authoring of generator pools (OpenRouter).

Strict boundary, so benchmark reproducibility never depends on a remote model:
LLMs are used OFFLINE, at pool-authoring time only. The surviving pool is
written to data/pools/*.json and COMMITTED; the deterministic generator
(stimuli.py) then only samples from committed pools using (config, seed).
Reruns never touch the API; the key's absence must never break generation.

Roles (supervisor rule: reviewer is same tier or STRONGER than the author,
preferably another lab):
  author   — proposes candidate event predicates
  reviewer — adversarially screens them for ordinal/temporal leakage

HARD RULE: every OpenRouter call goes through OPENROUTER_PROXY — no direct
connections. `_request()` raises if the proxy is unset rather than falling
back to a direct call.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

ENV_SEARCH_PATHS = [".env", "/workspace/manifolds/.env"]

# Words whose presence in a predicate leaks order/time and voids the stimulus.
ORDINAL_LEAK_HINTS = (
    "first last next then before after begin began start started end ended finish "
    "finished early late earlier later again resumed returned repeated replied "
    "responded answered echoed followed preceded continued stopped grew shrank "
    "increased decreased doubled halved aged old new young final initial"
).split()

AUTHOR_SYSTEM = """You write micro-events for a synthetic language-model benchmark.
Each item is a short PAST-TENSE verb phrase describing a self-contained, neutral,
physically generic event, e.g. "registered a wobbling", "absorbed a festrick",
"calibrated a spindle", "fused with a gondrel".

Hard constraints — violating any one makes the item useless:
- 2 to 5 words, past tense, lowercase, no trailing period
- NO words implying time, order, sequence, repetition, reaction, growth, decay,
  age, novelty, or completion (banned examples: {banned})
- no proper nouns, no real people/places/brands, no numbers, no pronouns
- emotionally neutral; no interaction with "another"/"the other" entity
- each phrase must stand alone with any subject: "the glemb <phrase>."

Return ONLY a JSON array of strings, no commentary."""

REVIEWER_SYSTEM = """You are the adversarial supervisor screening event predicates
for a benchmark that measures whether language models infer ORDER purely from
context. REJECT a predicate only for these specific poisons:

1. Temporal/ordinal vocabulary (first, next, then, again, final, early, ...).
2. Verbs whose meaning REQUIRES a prior event in a sequence: replied, responded,
   echoed, returned, resumed, repeated, followed, countered.
3. Growth/decay/aging/novelty semantics (grew, aged, renewed, decayed).
4. Explicit sequence-position words (began, initiated, concluded, finalized).
   Ordinary telic verbs are FINE: sealed, polished, welded, closed are all OK —
   completing a private action does not hint at list position.
5. PROPER nouns: named people, places, brands, products. Common nouns
   (spindle, membrane, vibration, bracket) are perfectly acceptable and expected.
6. Numbers or counts in any form.
7. Not a 2-5 word past-tense verb phrase, or contains pronouns, or references
   another entity in the list ("the other", "another").

Do not invent new rejection categories. When a case is genuinely ambiguous on
rules 1-4, reject; otherwise keep. Return ONLY a JSON array of objects:
[{"p": "<predicate>", "keep": true|false, "reason": "<short>"}]"""


def _load_env() -> dict:
    env = dict(os.environ)
    for path in ENV_SEARCH_PATHS:
        if os.path.exists(path):
            for line in open(path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env.setdefault(k, v)
            break
    return env


def _request(messages: list[dict], model: str, temperature: float, max_tokens: int = 4096) -> tuple[str, dict]:
    """One chat completion via OpenRouter, proxy-enforced. Returns (content, usage)."""
    env = _load_env()
    key = env.get("OPENROUTER_API_KEY") or ""
    proxy = (env.get("OPENROUTER_PROXY") or "").strip()
    base = env.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if not proxy:
        raise RuntimeError("OPENROUTER_PROXY is not set — direct OpenRouter calls are forbidden")

    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    ).encode()
    req = urllib.request.Request(
        base + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            resp = json.load(opener.open(req, timeout=180))
            if "error" in resp:
                raise RuntimeError(f"OpenRouter error: {resp['error']}")
            return resp["choices"][0]["message"]["content"], resp.get("usage", {})
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"OpenRouter call failed after retries: {last_err}")


def _extract_json(text: str):
    """Parse a JSON value from a model reply, tolerating code fences."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=0)
    return json.loads(text[start:])


AUTHOR_THEMES = [
    "mechanical/workshop actions",
    "optics and light (lenses, beams, reflections)",
    "fluids and gases (pouring, mixing, diffusing)",
    "textiles and soft materials (folding, weaving)",
    "sound and vibration (humming, resonating)",
    "geometry and spatial arrangement (tilting, nesting)",
    "temperature and state changes (warming, congealing)",
    "granular and powder handling (sifting, scattering)",
]


def author_event_predicates(
    n: int, model: str, existing: list[str] | None = None,
    temperature: float = 1.0, theme: str | None = None,
) -> tuple[list[str], dict]:
    """One authoring call proposing up to ~60 candidates. Returns (candidates, usage)."""
    system = AUTHOR_SYSTEM.format(banned=", ".join(ORDINAL_LEAK_HINTS[:18]) + ", ...")
    avoid = ""
    if existing:
        sample = existing[-80:]
        avoid = "\nAlready have (do NOT repeat or closely paraphrase):\n" + json.dumps(sample)
    theme_line = f" Draw mostly from this domain: {theme}." if theme else ""
    user = f"Propose {min(n, 60)} new candidate predicates.{theme_line}{avoid}"
    content, usage = _request(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=model, temperature=temperature,
    )
    cands = [str(p).strip().lower().rstrip(".") for p in _extract_json(content)]
    return [c for c in cands if 2 <= len(c.split()) <= 5], usage


def review_pool(candidates: list[str], model: str) -> tuple[list[str], list[dict], dict]:
    """Supervisor screen. Returns (kept, verdicts, usage)."""
    content, usage = _request(
        [
            {"role": "system", "content": REVIEWER_SYSTEM},
            {"role": "user", "content": json.dumps(candidates)},
        ],
        model=model, temperature=0.0,
    )
    verdicts = _extract_json(content)
    kept = [v["p"] for v in verdicts if v.get("keep")]
    return kept, verdicts, usage


def cheap_local_screen(candidates: list[str]) -> list[str]:
    """Free pre-filter before spending reviewer tokens: obvious leak words."""
    banned = set(ORDINAL_LEAK_HINTS)
    return [c for c in candidates if not (set(re.findall(r"[a-z]+", c)) & banned)]
