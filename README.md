# In-Context Order Manifolds

**Do language models build an ordered geometric representation of a novel sequence presented in their context window — and does the quality of that geometry predict, and causally determine, how well they can reconstruct the order?**

> Status: 🚧 Stage 0 — scaffolding & pilot (July 2026). Interfaces are stabilizing; no results yet.

---

## The question

Three facts are established separately in the interpretability literature, and this project tests their untested intersection:

1. **Pretrained continua live on curved 1D manifolds.** Years, weekdays, months, and character counts lie along smooth low-dimensional curves in activation space that models use causally (Engels et al. 2024; Modell et al. 2025; Gurnee et al. 2026).
2. **Models reorganize representations to mirror in-context structure** — but this has only been shown for *graph adjacency*, not for *ordered, ranked* structure (Park et al. 2025), and there is evidence such representations may be encoded yet inert (Lepori et al. 2026).
3. **Models behaviorally fail at order reconstruction** when input order is manipulated, losing global consistency while keeping local order (Fatemi et al. 2024; Wongchamcharoen & Glasserman 2025).

Nobody has tested whether a clean **ordered 1D manifold** forms for a **novel in-context sequence**, whether its **per-example geometric quality** predicts ordering accuracy, and whether **steering along the manifold** causally rescues ordering behavior. That is what this repo does.

### Research tracks

| Track | Hypothesis | Test |
|---|---|---|
| **A — Formation & prediction** | Hidden states of in-context items form a 1D manifold whose coordinate tracks the latent order; per-example quality predicts per-question accuracy | Mixed-effects regression of accuracy on manifold quality, within condition × N cells, controlling attention entropy and position |
| **B — Reverse vs. shuffle dissociation** | Reversed presentation preserves a content-order manifold (read-out failure); shuffled presentation degrades it (formation failure) — *after removing the positional component* | Position/content decomposition of the recovered coordinate across Forward / Reverse / Shuffle; cross-prompt subspace patching |
| **C — Causal steering** | Interventions *along* the manifold's local tangent shift an item's represented rank coherently; matched-norm and in-subspace-orthogonal controls do not | Dose–response steering with the signed prediction that the item's answered rank drifts monotonically by whole positions |

### Design commitments (what makes this defensible)

- **Position ≠ order.** Pooled per-item vectors always encode presentation position, so Forward/Reverse geometry is confounded by construction. Every geometric readout reports the full 2×2 — correlation of the recovered coordinate with latent rank *and* presentation position, plus both partials. **Shuffle is the identification condition**: content-order structure only counts if it survives controlling for position.
- **A semantic-scaffolding gradient.** Three stimulus families with identical latent orders: **dated** (fixed-width fictional dates), **tagged** (numeric rank tags), and **relational** (only adjacent "X before Y" statements — the global order exists solely via in-context transitive closure). Dates and tags let a model reuse its pretrained number-line; the relational family is the genuinely in-context case.
- **Nonce entities everywhere** — no order is recoverable from pretraining.
- **Identical question sets across conditions**; only card order varies. Five question families (full reconstruction, distance-stratified pairwise, adjacency, rank, anchored span), each targeting a distinct geometric prediction.
- **Geometry needs points.** N = 16–48 items for manifold metrics; topology (persistent homology, intrinsic dimension) only at N ≥ 24. Behavioral primaries are Kendall τ and distance-binned pairwise accuracy (exact match floors early and is secondary).
- **Reliability before regression.** The per-example quality score's split-half reliability is measured and reported; layer selection happens on a held-out split of latent orders.

---

## Repository layout

```
├── configs/
│   ├── models.yaml            # model roster + dtypes (V100-safe: fp16, eager attention)
│   ├── generation.yaml        # families, conditions, N grid, counterbalancing
│   └── experiments/           # one YAML per experiment = one results table
├── src/icom/
│   ├── generator/             # stimuli + questions, deterministic from (config, seed)
│   │   ├── entities.py        #   nonce entity vocabulary, per-tokenizer sanity checks
│   │   ├── stimuli.py         #   3 families × 4 conditions, fixed token budgets
│   │   ├── questions.py       #   5 question families with deterministic answer keys
│   │   └── schemas.py         #   dataclasses: the single source of truth for records
│   ├── battery/               # behavioral evaluation (HF runner, prefix KV-cache reuse)
│   │   ├── client.py          #   encode card-block once, answer all questions from cached KV
│   │   ├── scoring.py         #   parsers, exact match, tau, distance bins, logit scoring
│   │   └── run.py
│   ├── extraction/            # activation capture (HF hooks), pooling on the fly
│   │   ├── hooks.py           #   residual-stream capture; never dumps full [L×T×D]
│   │   ├── pooling.py         #   name / marker / last-token / card-mean schemes
│   │   └── run.py
│   ├── geometry/              # the measurement core
│   │   ├── curve.py           #   PCA + principal curve → arc-length coordinate
│   │   ├── quality.py         #   the 2×2 readout: ρ vs latent & position, + partials
│   │   ├── position.py        #   positional-subspace estimation and projection-out
│   │   ├── topology.py        #   persistent homology / TwoNN (N ≥ 24 only)
│   │   └── reliability.py     #   split-half reliability of quality scores
│   ├── steering/
│   │   ├── tangent.py         #   cross-stimulus (Procrustes-aligned) tangent estimation
│   │   └── intervene.py       #   along / matched-norm / in-subspace-orthogonal hooks
│   ├── stats/
│   │   └── models.py          #   preregistered mixed-effects specifications
│   └── utils/
├── scripts/                   # thin CLI entry points for the four pipeline steps
│   ├── generate_dataset.py
│   ├── run_battery.py
│   ├── extract_activations.py
│   ├── analyze_geometry.py
│   └── infra/                 # docker helpers for the GPU workers (env-parameterized)
├── tests/
└── notebooks/                 # exploratory only; nothing load-bearing lives here
```

The pipeline is four idempotent steps, each keyed by content hash so interrupted runs resume per-stimulus:

```bash
python scripts/generate_dataset.py   --config configs/generation.yaml --out data/
python scripts/run_battery.py        --config configs/experiments/pilot.yaml   # → results/*.parquet
python scripts/extract_activations.py --config configs/experiments/pilot.yaml  # → acts/*.npz (pooled only)
python scripts/analyze_geometry.py   --config configs/experiments/pilot.yaml   # → results/geometry.parquet
```

Every results row carries `git_sha, model, layer, pooling, family, condition, N, seed`.

## Setup

```bash
pip install -e ".[dev]"   # generator + geometry + stats (CPU-only, runs anywhere)
pip install -e ".[gpu]"   # + extraction and battery (on GPU workers)
```

**GPU workers.** The reference fleet is two 8× V100-SXM2-32GB nodes. V100 (SM 7.0) implies hard constraints baked into the configs: **fp16 only** (no bfloat16; every model passes an fp16-vs-fp32 logit-KL smoke test before entering the grid), **eager/SDPA attention** (no FlashAttention), and **torch < 2.7 / cu126 wheels** (PyTorch drops Volta from cu128+ builds). Modern vLLM has no Volta support at all, so the behavioral battery runs on plain transformers with manual prefix KV-cache reuse: the card-block prefix is encoded once per stimulus and all ~25 questions are answered from the cached `past_key_values`. Activation extraction uses plain HF forward hooks and stores only pooled `[N_items × layers × D]` fp16 tensors (~MBs per stimulus).

**Model roster** (`configs/models.yaml`):
- **OLMo-3-7B** (Base + Instruct) — flagship: standard transformer, Apache 2.0, ungated, **fully open training data** (nonce-novelty verifiable) and **intermediate checkpoints at every stage** (developmental Track E for free); OLMo-3-32B for scale-up confirmation with activations, across 3 GPUs.
- **Qwen3 dense 1.7B / 4B / 8B / 14B** — the scale axis: standard GQA+RoPE attention (the hybrid designs only begin with Qwen3-Next/3.5), Apache 2.0, ungated; battery runs with `enable_thinking=false`.
- **SmolLM3-3B** (optional) — fully open, NoPE-interleaved layers: a natural contrast case for the position-confound analyses.
- **Exploratory architecture contrast**: Olmo-Hybrid-7B and Qwen3.5-9B (hybrid Gated-DeltaNet families) — only if their kernels run on SM 7.0.
- Excluded: Gemma-3 (bf16-trained, fp16-overflow history, gated), Llama-3/4 (gated; 4.x MoE), Pythia/Qwen2.5 (superseded by OLMo-3 and Qwen3 respectively).

## Roadmap

- [x] Research plan, critique pass, hardware audit
- [ ] **Stage 0** — generator (3 families × 4 conditions), analysis library, end-to-end pilot on OLMo-3-7B-Instruct + Qwen3-4B
- [ ] **Stage 1** — identifiability test (does content-order signal survive position removal in Shuffle?) + behavioral replication of the sorted-vs-shuffled gap
- [ ] **Stage 2** — Tracks A + B at full grid (3 models × 3 families × 4 conditions × 3 N × ~300 stimuli)
- [ ] **Stage 3** — Track C steering with dose–response controls
- [ ] **Stage 4** — scaling checks (Qwen3-14B; OLMo-3-32B with activations across 3 GPUs), write-up

## References (core)

- Engels et al., *Not All Language Model Features Are One-Dimensionally Linear*, [arXiv:2405.14860](https://arxiv.org/abs/2405.14860)
- Park et al., *In-Context Learning of Representations*, [arXiv:2501.00070](https://arxiv.org/abs/2501.00070)
- Modell, Rubin-Delanchy & Whiteley, *The Origins of Representation Manifolds in LLMs*, [arXiv:2505.18235](https://arxiv.org/abs/2505.18235)
- Gurnee et al., *When Models Manipulate Manifolds*, [arXiv:2601.04480](https://arxiv.org/abs/2601.04480)
- Lepori et al., *Language Models Struggle to Use Representations Learned In-Context*, [arXiv:2602.04212](https://arxiv.org/abs/2602.04212)
- Wurgaft et al., *Manifold Steering Reveals the Shared Geometry of Representation and Behavior*, [arXiv:2605.05115](https://arxiv.org/abs/2605.05115)
- Fatemi et al., *Test of Time*, [arXiv:2406.09170](https://arxiv.org/abs/2406.09170)
- Wongchamcharoen & Glasserman, *Do LLMs Understand Chronology?*, [arXiv:2511.14214](https://arxiv.org/abs/2511.14214)
- Berglund et al., *The Reversal Curse*, [arXiv:2309.12288](https://arxiv.org/abs/2309.12288) — the in-weights contrast case; this project operates entirely in-context, where the curse does not apply

## Citation

If you build on this work before a paper is out, please cite the repository:

```bibtex
@misc{galyukshev2026icom,
  author = {Galyukshev, Ilya},
  title  = {In-Context Order Manifolds},
  year   = {2026},
  url    = {https://github.com/IlyaGalyukshev/in-context-order-manifolds}
}
```

## License

MIT
