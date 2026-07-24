# In-Context Order Manifolds

**Do language models build a persistent, entity-indexed geometric representation — a manifold — of a novel relational structure given purely in their context window, and if so, at which layers?**

> Status: 🚧 instrument built and audited; the decisive activation run awaits GPU. No headline results yet.

---

## The question

When you define a novel order over nonce entities *in-context* (only via stated pairwise relations), does the model construct an internal geometric code, indexed by entity, whose coordinate mirrors that order — a **manifold** — or does it compute order **on demand at the query**, with no persistent per-entity representation? And where, in depth, would such a manifold live?

This matters because pretrained continua (years, weekdays, character counts) are known to live on curved low-dimensional manifolds the model uses causally — but that is *retrieval of a learned feature*, not *in-context construction of a novel one*. The in-context case is open, and easy to get wrong: a naive probe reports a "manifold" that is really a surface artifact.

## Why this is hard — the confound that eats naive designs

If you state a novel order as a chain of "A before B" cards and probe the entity token for rank, you will find a signal — but it can be an **endpoint/role artifact**, not order geometry: in a linear chain the first item is only ever a subject and mentioned once, the last only an object, so *rank correlates with syntactic role and mention frequency*. A rank probe then reads role, not order. (We confirmed this: rank decodes overall but **collapses to null among interior items** that share role and frequency.)

The whole benchmark is built to remove this at the **data level**, so any surviving signal is genuine integrated-order representation.

## The design — Balanced Comparability Sets (BCS)

Each stimulus states a **redundant, degree-regular** set of order comparisons over nonce entities:

- **Degree-regular comparison graph containing the Hamiltonian path** → the order is *uniquely determined* (transitive closure) yet every entity is mentioned exactly `d` times ⇒ **mention frequency is rank-invariant**.
- **Eulerian-balanced phrasing** ("A is smaller than B" vs "B is larger than A" with a 50/50 orientation) ⇒ **syntactic first-position is exactly 0.5 for every entity, independent of rank**.
- **Non-adjacent edges** ⇒ the order cannot be read off a single card by local chaining; it must be integrated.
- **Rank-decorrelated card order** (the *shuffle* condition) and a **readout roster** appended after all cards give each entity a clean, post-integration read position (so *"which layer"* is not confounded by *"which mention"*).

Confound controls are **first-class metrics**, not afterthoughts:

- **Interior-only rank decoding** (ranks 3..N−2, identical role/frequency) is the PRIMARY geometry number, against a within-stimulus permutation null.
- **Coherence-null twins** (relations forming a cycle → no valid order) must decode at chance, or we are reading card statistics.
- A **data-level confound audit** proves the fix without a GPU: rank decodable from role features drops from ≈0.32 (linear chain) to ≈0 (BCS).

## What it can measure

- **Semantics gradient** — the same abstract task under a symbolic relation ("zibs", transitivity declared in-context) vs meaningful comparatives ("smaller/louder") → does order representation need meaning?
- **Difficulty ladder** — easy (short-range padding, locally chainable) vs hard (long-range padding, forces global integration), on **content-matched** stimuli → guarantees a solvable regime and separates competence from geometry.
- **Non-total structures** — partial orders (two incomparable chains; tests whether the model invents a spurious total order) and 2-D structures (two independent orders; tests whether intrinsic dimension mirrors the structure).
- **Behavior ⟂ geometry** — a behavioral battery (reconstruction, distance-stratified pairwise, rank, order-query with an incomparability option) run at the same stimuli, so geometry is only interpreted where behavior is above chance.

## Repository layout

```
├── configs/
│   ├── models.yaml            # roster + dtypes (V100-safe: fp16, eager attention)
│   └── generation.yaml        # entity/predicate pools + offline LLM pool-authoring config
├── data/pools/                # committed nonce-entity & event-predicate pools (deterministic inputs)
├── src/icom/
│   ├── generator/
│   │   ├── bcs.py             #   Balanced Comparability Sets: total / partial-order / 2-D
│   │   ├── bcs_questions.py   #   reconstruction, swap-paired pairwise, rank, order-query
│   │   ├── entities.py        #   nonce vocabulary + wordfreq + per-tokenizer screen
│   │   └── llm_assist.py      #   offline predicate-pool authoring via OpenRouter (proxy-only)
│   ├── extraction/            # residual-stream capture (HF hooks); pools per entity on the fly
│   │   ├── hooks.py           #   one forward pass/stimulus; never dumps full [L×T×D]
│   │   └── pooling.py         #   readout(roster) / name / last-token / card-mean schemes
│   ├── battery/               # behavioral eval (HF runner) — client.py + scoring.py
│   └── utils/seeding.py       # deterministic (config, seed) → identical artifacts
├── scripts/
│   ├── generate_bcs.py        # build a dataset (stimuli + questions + coherence-null twins)
│   ├── build_entity_pool.py   # nonce vocabulary; author_pools.py — predicate pool (offline LLM)
│   ├── run_battery.py         # behavior gate (must precede geometry)
│   ├── extract_activations.py # pooled [N × layers × D] per stimulus
│   ├── probe_interior.py      # PRIMARY: interior-only rank probe per layer + perm/coherence nulls
│   ├── probe_rank.py          # all-ranks layer sweep (all-vs-interior comparison)
│   ├── audit_confounds.py / adv_confound_audit.py / adv_decode.py  # data-level confound audits
│   ├── deploy_knockout.py / steer_rank.py   # deployment / causal tests
│   └── smoke_v100.py          # per-model fp16 admission test (V100)
└── tests/test_bcs.py          # 32 generation invariants (all six confound gates, per config)
```

Every dataset is deterministic from `(config, seed)`; the per-stimulus path never calls an API (LLMs only author the committed predicate pool, offline).

## Pipeline

```bash
pip install -e ".[dev]"     # generator + analysis (CPU, runs anywhere)
pip install -e ".[gpu]"     # + extraction and battery (GPU workers)

python scripts/generate_bcs.py --out data/bcs --families s0_zib,s1_size,s1_loud \
    --n-grid 7,9,12,16 --per-cell 120 --difficulty both --structures
python scripts/run_battery.py        --model <m> --stimuli data/bcs/stimuli.jsonl ...   # behavior gate
python scripts/extract_activations.py --model <m> --stimuli data/bcs/stimuli.jsonl ...   # readout locus
python scripts/probe_interior.py --acts <acts> --scheme readout --family s1_size --condition shuffle
```

**V100 constraints** (baked into configs): fp16 only (per-model fp16-vs-fp32 logit-KL smoke test), eager/SDPA attention, `torch<2.7`/cu126, `transformers>=4.57`. The battery is a plain-transformers runner (modern vLLM has no Volta support).

## The decisive measurement

Interior-only rank decodability at the readout locus, per layer, in a *solvable* cell, against the permutation and coherence nulls. One curve decides **manifold vs query-local computation** and answers **at which layers** — everything else refines it. Any claim must replicate on ≥3 models.

## License
MIT
