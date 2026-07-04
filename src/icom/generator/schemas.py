"""Dataclasses shared across the whole pipeline — the single source of truth.

Design invariants encoded here:
- A stimulus always records BOTH latent rank and presentation slot per item,
  because the central analysis decomposes geometry into content-order vs.
  positional components (the two are confounded in Forward/Reverse by design).
- Questions are generated once per latent order and reused verbatim across
  all conditions of the same content: only card order differs, never questions.
- IDs are content hashes so that interrupted runs resume idempotently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


class StimulusFamily(str, Enum):
    """Semantic-scaffolding gradient, strongest to weakest pretrained support."""

    DATED = "dated"            # fixed-width fictional dates (full pretrained scaffold)
    TAGGED = "tagged"          # fixed-width numeric rank tags (weaker scaffold)
    RELATIONAL = "relational"  # only adjacent "X before Y" statements; order exists
    # solely via in-context transitive closure (flagship)


class Condition(str, Enum):
    FORWARD = "forward"          # presentation slot == latent rank (confounded; control only)
    REVERSE = "reverse"          # presentation slot == N+1-latent rank (still confounded)
    SHUFFLE = "shuffle"          # slot ⟂ rank — the identification condition for geometry
    POSITION_CONTROL = "posctl"  # sorted content, queried items forced mid-context


class QuestionFamily(str, Enum):
    RECONSTRUCTION = "reconstruction"  # global: full ordered list
    PAIRWISE = "pairwise"              # local, stratified by rank distance
    ADJACENCY = "adjacency"            # immediate successor
    RANK = "rank"                      # absolute coordinate readout
    SPAN = "span"                      # anchored sub-span (start / middle / end)


@dataclass(frozen=True)
class Card:
    """One event card. `latent_rank` is ground truth; `presentation_slot` is
    where the card actually appears in the prompt (1-indexed)."""

    entity: str
    text: str
    latent_rank: int
    presentation_slot: int


@dataclass
class Stimulus:
    family: StimulusFamily
    condition: Condition
    n_items: int
    seed: int
    cards: list[Card]
    prompt: str
    latent_order: list[str] = field(default_factory=list)  # entities, rank 1..N

    @property
    def stimulus_id(self) -> str:
        payload = json.dumps(
            [self.family, self.condition, self.n_items, self.seed, self.prompt],
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Question:
    stimulus_content_key: str  # shared across conditions of the same latent order
    family: QuestionFamily
    text: str
    answer_key: str | list[str]
    # Analysis metadata:
    rank_distance: int | None = None   # pairwise: |rank(A) - rank(B)|
    is_endpoint: bool | None = None    # adjacency/rank: anchors behave differently
    span_location: str | None = None   # span: start / middle / end


@dataclass
class BatteryRow:
    """One scored completion. Every row carries full provenance."""

    stimulus_id: str
    question_family: str
    correct: bool | None          # None on parse failure (logged, never scored wrong)
    score: float                  # graded metric (tau, set overlap, logit margin, ...)
    parse_failed: bool
    raw_completion: str
    model: str
    git_sha: str
    seed: int


@dataclass
class GeometryRow:
    """The 2x2 readout for one (stimulus, layer, pooling) cell.

    `q_content_partial` — Spearman rho(coordinate, latent rank | presentation slot) —
    is THE dependent variable. Raw rho vs latent order is reported but never
    interpreted alone: in Forward/Reverse it is confounded with position.
    """

    stimulus_id: str
    model: str
    layer: int
    pooling: str
    rho_latent: float             # coord vs latent rank (confounded in fwd/rev)
    rho_position: float           # coord vs presentation slot
    q_content_partial: float      # coord vs latent rank, controlling slot  ← primary
    q_position_partial: float     # coord vs slot, controlling latent rank
    quality_reliability: float    # split-half reliability of q_content_partial
    intrinsic_dim: float | None   # None for N < 24
    ph_summary: str | None        # None for N < 24
    git_sha: str = ""
