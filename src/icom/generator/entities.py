"""Nonce entity vocabulary.

Names are built from curated onset/nucleus/coda inventories (glemb-style:
pronounceable, obviously non-English-but-English-like). Deterministic from
seed. Before a dataset is generated, the vocabulary must pass
`check_tokenization` on EVERY tokenizer in the roster (1–3 tokens per name);
the generator refuses names that any roster tokenizer explodes.
"""

from __future__ import annotations

import numpy as np

ONSETS = [
    "gl", "dr", "qu", "sn", "pl", "gr", "fl", "br", "cr", "tr",
    "sp", "st", "sk", "bl", "cl", "fr", "pr", "sl", "sm", "thr",
]
NUCLEI = ["e", "a", "o", "i", "u", "ee", "oa", "ie"]
CODAS = [
    "mb", "ne", "nch", "nnel", "rvic", "mth", "ndrel", "strick",
    "lb", "rn", "sk", "ft", "mp", "nd", "rl", "pt", "x", "zzle",
]

# Accidental real words / near-collisions to exclude (extend as found).
BLOCKLIST = {
    "grand", "brand", "stand", "trend", "spend", "blend", "friend",
    "plane", "crane", "drone", "stone", "brine", "spine", "smile",
    "fleet", "sleet", "greet", "street", "flask", "brisk", "crisp",
    "trunk", "drink", "blink", "plank", "prank", "clamp", "stamp",
    "quest", "crest", "frost", "trust", "twist", "quilt", "spelt",
    "smart", "start", "sport", "short", "chart", "smelt", "spurn",
}


def _is_real_word(name: str) -> bool:
    """Frequency-based screen against real English words (wordfreq required).

    The hand-curated BLOCKLIST proved insufficient in pilot ('stanch' slipped
    through) — a real word carries pretrained associations, which poisons the
    novelty guarantee. Zipf > 1.5 ≈ appears in real text at all.
    """
    from wordfreq import zipf_frequency

    return zipf_frequency(name, "en") > 1.5


def build_vocabulary(seed: int, size: int = 500) -> list[str]:
    """Generate `size` unique nonce names, deterministic from `seed`."""
    rng = np.random.default_rng(seed)
    seen: set[str] = set()
    out: list[str] = []
    max_attempts = size * 200
    for _ in range(max_attempts):
        if len(out) >= size:
            break
        name = rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
        if name in seen or name in BLOCKLIST or len(name) < 4 or len(name) > 10:
            continue
        if _is_real_word(name):
            continue
        seen.add(name)
        out.append(name)
    if len(out) < size:
        raise ValueError(f"could only build {len(out)}/{size} names — extend inventories")
    return out


def check_tokenization(vocab: list[str], tokenizer, max_tokens: int = 3) -> list[str]:
    """Subset of `vocab` that stays within `max_tokens` pieces for `tokenizer`.

    Checked with a leading space ("the glemb") — the in-context form — since
    many BPE tokenizers split differently at word boundaries.
    """
    ok = []
    for w in vocab:
        n_bare = len(tokenizer(w, add_special_tokens=False)["input_ids"])
        n_spaced = len(tokenizer(" " + w, add_special_tokens=False)["input_ids"])
        if max(n_bare, n_spaced) <= max_tokens:
            ok.append(w)
    return ok
