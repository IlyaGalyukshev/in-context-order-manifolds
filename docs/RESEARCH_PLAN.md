# Research plan — the wide sweep, the decision tree, and the manifold pictures

**Question.** Does a language model build a persistent, entity-indexed geometric
representation — a *manifold* — of a novel relational structure given purely
in-context, and if so at which layers, and how does that depend on architecture,
query type, difficulty, length, and semantics?

The design is *extract wide, probe exhaustively offline, then select from the
results*. Extraction (GPU) is paid once per (model × stimulus × locus); the whole
probe catalog is then pure-CPU numpy over stored `[N × (L+1) × D]` fp16 tensors,
so every probe family × locus × layer × condition is run and the surviving
signals are chosen afterward.

## Positioning (see [`related_work.md`](related_work.md))
Prior work separately shows (i) novel in-context structure reorganizes geometry
(Park et al. 2501.00070; Kowalyshyn et al. 2605.08405), (ii) ordinal/magnitude
quantities live on *nonlinear* manifolds — helix/circle/log/place-cell
(Kantamneni & Tegmark 2502.00873; Engels et al. 2405.14860; Levy & Geva
2410.11781; Gurnee et al. 2601.04480), and (iii) a pretrained magnitude/space axis
exists and is steerable (Tehenan et al. 2506.02996; Heinzerling & Inui 2403.10381;
Hu et al. 2602.06843). The two nearest neighbors: **Singh 2606.01269** (an emergent
rank line + a reusable ordinal axis in a frozen LLM) and **Bassi & Tomar 2607.04167**
(ordinal manifolds *degrade in the cross-position-integration regime*). This
project targets the intersection none covers — a confound-clean, nonce-entity,
**in-context-constructed** order, decoded **interior-only** against a permutation
null, characterized for **nonlinearity**, localized in **depth** across **≥3
architectures**, with a causal test of pretrained-axis reuse. Two implications:
**nonlinear characterization is co-primary** (the manifold is likely curved), and
BCS is the **adjudicator of exactly the integration regime** where ordinal-manifold
coherence is reported to break down — a clean interior manifold there contradicts
Bassi & Tomar (surprising positive); a collapse confirms and cleans them (strong
negative). Either outcome is publishable.

## The central fork
- **MANIFOLD** — a persistent, entity-indexed geometric code, present *after
  reading the cards*, whose coordinate mirrors the structure.
- **QUERY-LOCAL COMPUTATION** — order computed on demand at the question, with no
  persistent entity representation.

Interior-only rank decodability at the readout locus, per layer, against the
permutation and coherence nulls, is the arbiter. (A naive design finds a
"manifold" that is an endpoint/role artifact — see the README's confound section
and `audit_confounds.py`; the interior-only decode is what survives that.)

## The sharpest lever — architecture makes the fork mechanistic
This is not "more models". The architecture changes what *query-local* even
means:
- **Dense, full attention** (Qwen3, OLMo-3). At the question the model can
  re-attend to every card ⇒ order can be recomputed query-locally; a persistent
  manifold is *optional*.
- **Linear-attention / SSM** (Falcon-Mamba, OLMo-Hybrid, RWKV). The cards are
  gone from the state at query time (bounded recurrent state, no per-token KV to
  re-read) ⇒ the order *must* be compressed into a persistent state as the cards
  stream in, or it is lost. These are *forced* onto the manifold side — or they
  fail at large N when the state bottlenecks.
- **MoE** (OLMoE, Qwen3-30B-A3B). Is the order code carried in the
  routing-invariant residual stream, or fragmented across experts (does
  decodability depend on which expert fired at the readout token)?

**Prediction ladder:** SSM ≥ hybrid ≥ dense in interior decodability at matched
behavior; SSM order degrades with N earlier (state bottleneck) while dense holds
via re-attention. The most informative single outcome is a **dissociation** —
e.g. SSM decodes an interior manifold while dense does not — which would show the
manifold is real but built only when the architecture forces it (query-local
being the dense shortcut).

## The sweep matrix
| Axis | Levels |
|---|---|
| Architecture | dense (Qwen3 1.7/4/8/14B, OLMo-3-7B, SmolLM3-3B) · MoE (OLMoE-1B-7B, Qwen3-30B-A3B) · SSM/linear/hybrid (Falcon-Mamba-7B, OLMo-Hybrid-7B; expl: Zamba2, RWKV-7, Qwen3.5) |
| Semantics | S0 symbolic ("zibs") · S1 comparative (size / loud / heat) |
| Structure | total order · partial order (2 chains) · 2-D grid · **cyclic/modular ring** (nonlinear litmus — a valid single cycle, distinct from the invalid-cycle null) |
| N | 7, 9, 12, 16 (+ 24 stress for the SSM state-bottleneck test) |
| Difficulty | easy (near padding, locally chainable) · hard (far padding, forces global integration) — content-matched |
| Condition | shuffle (identification) · forward (ceiling/control) · coherence-null (cycle → chance) |
| Query category | reconstruction · pairwise (rank-distance stratified) · rank · order-query (incomparability) · betweenness · successor/predecessor · count-between · comparative-distance · extremes (endpoint-diagnostic) |
| Read locus | readout (roster, primary) · name · last_token · card_mean · marker · query-token (question appended — the query-local locus) |
| Layer | all layers (full depth profile), reported as a fraction of depth |
| Probe family | the 10 below |

## New experiments from the 2026 literature
- **Nonlinear characterization (co-primary).** Because ordinal/magnitude codes are
  reported curved (helix/circle/log/place-cell), the manifold's *shape* is a
  headline, not a footnote: principal-curve residual curvature, linear-vs-
  nonlinear-vs-geodesic (Isomap) decode gap, the Engels *irreducibility/separability*
  test (curvature is genuine, not a sum of linear parts), global PCA/RSA vs local
  probe, and an explained-variance + continuity check (high R² ≠ smooth manifold).
- **Cyclic/modular structure (litmus).** A new stimulus family: a valid single
  cycle over nonce entities (modular successor / cyclic-betweenness), predicting a
  *circular* manifold recoverable by circle-fit + circulant-RDM + nonlinear probe
  where a linear probe fails. The sharpest test that an in-context order can be
  nonlinear.
- **Pretrained-axis reuse (flagship extension).** Does the novel in-context order
  ride on a pretrained magnitude/size/space axis? Extract the source axis from a
  *comparative* ("which is bigger") contrast (not digit identity — per-digit
  landmine), align via cosine/CCA/cross-domain transfer, and *causally ablate* it
  to see if BCS ordering breaks — with **S0 (arbitrary "zib" order, zero magnitude
  lexicon)** as the clean abstract target no prior work has.
- **Difficulty ladder as adjudication.** easy (locally chainable) vs hard (forces
  integration) is now a direct test of Bassi & Tomar's local-vs-integration
  boundary — the confound-clean arbiter of where ordinal-manifold coherence holds.

## Phased protocol

**Phase A — Behavior gate (precedes all geometry).** Battery on every
architecture, per (family × structure × N × difficulty × condition).
Integration-requiring metrics only: multi-hop pairwise (rank-distance ≥ 2,
interior-only, swap-averaged; chance 0.5), reconstruction Kendall τ (chance 0),
and the metric categories (betweenness, count-between, comparative-distance).
Locate the *solvable regime* per architecture — geometry is interpreted only
where behavior is above chance. If nothing is solvable even at easy N=7–9 for a
model, that is a behavioral-limit result for it (re-check with a larger sibling).

**Phase B — Wide extraction.** One hooked forward pass per stimulus (card block +
roster, *no* question), all layers, all loci pooled on the fly →
`[N_entities × (L+1) × D]` fp16 (never the full `[L×T×D]`). Extract solvable
cells + their coherence-null twins + a matched shuffle/forward pair, every
architecture. A second pass per (stimulus × sampled question) with the question
appended pools only the **query-token** locus (the Result-C probe).

**Phase C — Exhaustive offline probe sweep (pure CPU).** Run the catalog over
stored activations:
1. **Interior-only linear ridge rank probe** (ranks 3..N−2) + within-stimulus
   permutation null (≥200) + coherence-null gate. **Primary.**
2. **Per-rank decodability profile** — endpoints only vs interior too.
3. **Nonlinear (small MLP) probe** — curvature signature (linear vs nonlinear gap).
4. **Representational geometry (RSA)** — Spearman of pairwise activation distance
   vs |rank difference|, interior-only (metric, not just linear-separable order).
5. **Intrinsic dimension** (TwoNN / MLE) per layer — ~1 (single order), ~2
   (grid/partial), or high (no manifold).
6. **Cross-condition transfer** — train shuffle / test forward (position-invariant
   code vs position-tied artifact).
7. **Cross-N generalization** — train N=9 / test N=12 (reusable coordinate scale).
8. **Cross-family transfer** — S0↔S1, size↔loud (shared order subspace).
9. **Cross-architecture geometry** — Procrustes/CCA alignment of the rank subspace
   across models (same geometry across architectures?).
10. **Depth dynamics** — onset (first layer > null), peak, contiguous significant
    band; the "which layers" answer, per model as a fraction of depth.

**Phase D — Selection & synthesis.** Keep only signals surviving ALL of:
interior-only > permutation null, coherence-null ≈ chance, and cross-condition
transfer. Assign an outcome per model (tree below) and build the headline
artifacts, including the manifold visualizations.

**Phase E — Deep dives (where Phase D says manifold).** Which-layers profile;
dimensionality/structure (partial → 2 components / ID≈2; grid → both axes
disentangled); deployment — does the decoded coordinate predict the model's *own*
answer beyond ground truth (intervention-free), then causal knockout + steering
along an interior-derived direction (dose-response vs matched-random vs
in-subspace-orthogonal).

**Phase F — Scale / replication.** Any manifold/layer/architecture claim must
replicate on ≥3 models spanning ≥2 architecture classes. Developmental: OLMo-3
and OLMoE checkpoints — when in training does the manifold appear?

## Deliverables — the 2D/3D manifold visualizations
The headline output is a **picture of the manifold**, backed by the quantitative
probe. For the interior entities at the readout locus, at the significant layers:

- **Projection.** Top-3 PCs (linear) and a nonlinear embedding — Isomap /
  spectral / LLE from `sklearn.manifold` (no extra dependency), optionally
  UMAP / PHATE — to 2-D and interactive 3-D.
- **Total order (1-D structure).** Scatter colored by latent rank → expect a
  smooth curved 1-D arc, monotone in rank. This *is* the manifold.
- **2-D grid.** Color by each axis (size, loud) separately → expect a 2-D sheet
  with the two colors varying along two directions (disentanglement).
- **Partial order.** Expect two arcs / two components, not one global line.
- **Per-layer contact sheet.** The same projection at every layer → watch the
  manifold *form and dissolve* with depth (the visual "which layers").
- **Architecture panel.** Dense vs MoE vs SSM projections side by side → the
  dissociation, if any, made visible.
- **Control panels, always adjacent.** The coherence-null twin (should be a
  blob, not an arc) and shuffle-vs-forward, in the *same* figure.

**Honesty rule for the pictures.** A projection is illustrative, never the claim:
a curved arc in PCA can arise from a confound, and low-dimensional embeddings
impose structure. Every manifold figure ships next to its null panel and is
subordinate to the interior-only decode vs the permutation/coherence nulls.
Planned tool: `scripts/visualize_manifold.py` (static PNG grids via matplotlib +
interactive 3-D HTML via plotly).

## Decision tree
| Outcome | Meaning | Next |
|---|---|---|
| **A** manifold forms (mid-late layers) + used | real & causal entity manifold | dimensionality depth, scale, developmental |
| **B** forms but inert | encoded, not deployed (Lepori-aligned) | locate the read-out bottleneck |
| **C** no manifold despite solving | order is query-local computation | query-token / last-token loci; attention-pattern analysis |
| **D** unsolvable everywhere | can't integrate novel in-context relations | scale models; add ICL worked examples |

Secondary axes refine A/B: semantics (needs meaning?), difficulty (weaker/later
on hard → integration depth scales with difficulty), length (weakens with N while
endpoints persist → a mechanistic account of the behavioral length collapse).

## Prioritization
Run the **decisive trio first** — one per class, small-N easy total-order:
Qwen3-4B (dense) · OLMoE-1B-7B (MoE) · Falcon-Mamba-7B (SSM). That trio alone
tests the architecture prediction and de-risks the SSM/MoE extraction path
(SM 7.0 kernels, hidden-state exposure, router logits). Then the same-lab
dense-vs-hybrid pair (OLMo-3-7B vs OLMo-Hybrid-7B) isolates architecture cleanly.
Then fan out scale, structures, and the N=24 stress.

## The single most important number
Interior-only rank decodability at the readout locus in the solvable easy cell,
per layer, vs the permutation and coherence nulls — for a dense model *and* an
SSM model. Their comparison decides A/B vs C, answers "which layers", and tests
the architecture lever in one figure. The manifold picture visualizes it;
everything else in the sweep refines it.

## Implementation status
Built and verified on CPU:
- BCS generator + full question battery incl. the metric families
  (betweenness/successor/count-between/comparative-distance/extremes);
  `tests/test_bcs.py` = 35 invariants.
- Offline probe catalog `src/icom/probes/` (linear interior + perm null,
  per-rank MAE, nonlinear/curvature, RSA, TwoNN intrinsic dim, transfer, depth)
  driven by `scripts/probe_sweep.py`; the 2D/3D visualizer
  `scripts/visualize_manifold.py`; `tests/test_probes.py` = 9 tests on synthetic
  activations (interior survives on a planted manifold, collapses on noise;
  TwoNN recovers dimension; nonlinear beats linear on a curved manifold;
  transfer across N; projection axis tracks rank). `benchmark_strata.py` +
  committed strata.
- `extract_activations.py` stores structure coordinates (grid axes / partial
  chains) for Phase E.

Remaining, GPU-side (implement at bring-up where they can be smoke-tested):
the query-token extraction pass (Result-C locus) and the SSM/MoE `smoke_v100.py`
extension (forward + hidden-state exposure + router logits). No headline results
yet — extraction awaits a card.
