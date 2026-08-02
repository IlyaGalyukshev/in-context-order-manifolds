# Related work

Annotated, link-verified bibliography for *in-context order manifolds*, grouped by
theme. Geometry tag per entry: **linear · log · circular/per-digit ·
helical/sinusoidal · curved-manifold · polytope · behavioral · method**. Emphasis
on nonlinear structure and on methods to detect and defend it. A short positioning
statement closes the file.

> Scope of this project: does a language model build a *persistent, entity-indexed,
> possibly nonlinear geometric manifold* of a **novel in-context relational order
> over nonce entities**, versus computing order query-locally; at which layers;
> across architectures; and does such an order reuse a pretrained magnitude/space
> axis. Measured interior-only against a within-stimulus permutation null (BCS).

## Nonlinear / curved / circular / helical geometry (and how to detect it)
- **Engels et al., "Not All LM Features Are One-Dimensionally Linear"** — [2405.14860](https://arxiv.org/abs/2405.14860). *circular/multi-dim.* SAE + geodesic clustering + an irreducibility/separability test find circular day/month features, causally used in modular arithmetic. The reference method for proving a feature is genuinely multi-dimensional/curved.
- **Kantamneni & Tegmark, "LMs Use Trigonometry to Do Addition"** — [2502.00873](https://arxiv.org/abs/2502.00873). *helical.* Numbers on a generalized helix (linear × periodic); a "Clock" algorithm; causally verified.
- **Zhong et al., "The Clock and the Pizza"** — [2306.17844](https://arxiv.org/abs/2306.17844). *circular.* Modular addition → circular embeddings; two distinct algorithms across seeds (a caution for replication).
- **Modell, Rubin-Delanchy & Whiteley, "The Origins of Representation Manifolds in LLMs"** — [2505.18235](https://arxiv.org/abs/2505.18235). *curved-manifold (theory).* Features are manifolds; cosine similarity encodes on-manifold geodesic distance — why linear probes can under-decode a curved variable.
- **"Cylindrical Representation Hypothesis for LM Steering"** — [2605.01844](https://arxiv.org/abs/2605.01844). *curved (cylinder).* Non-orthogonal axis + angular sectors; explains variable steering outcomes.
- **Li et al., "The Geometry of Concepts: SAE Feature Structure"** — [2410.19750](https://arxiv.org/abs/2410.19750). *mixed.* Crystal/lobe/galaxy structure; LDA distractor removal; eigenspectrum by layer.
- **Park et al., "Geometry of Categorical and Hierarchical Concepts"** — [2406.01506](https://arxiv.org/abs/2406.01506). *polytope/simplex.* Categories = simplices; hierarchy = orthogonality — vocabulary for non-total structures.
- **Elhage et al., "Toy Models of Superposition"** — [2209.10652](https://arxiv.org/abs/2209.10652). *polytope.* Feature packing as geometric configurations under sparsity.
- **Štefánik et al., "Universal Representations of Numbers"** — [2510.26285](https://arxiv.org/abs/2510.26285); **Kadlčík et al., "Remarkably Accurate Representations of Numbers"** — [2506.08966](https://arxiv.org/abs/2506.08966). *sinusoidal.* Model families converge on periodic number codes.
- **Davies et al., "LMs Do Not Embed Numbers Continuously"** — [2510.08009](https://arxiv.org/abs/2510.08009). *discrete/per-digit.* High linear R² coexists with a non-continuous code (high accuracy ≠ smooth manifold).

Detection methods: intrinsic dimension — **Facco et al., TwoNN** ([Sci. Rep. 2017](https://www.nature.com/articles/s41598-017-11873-y)), **Ansuini et al.** ([1905.12784](https://arxiv.org/abs/1905.12784)), **Valeriani et al.** ([2302.00294](https://arxiv.org/abs/2302.00294)); **RSA** — Kriegeskorte et al. ([Frontiers 2008](https://www.frontiersin.org/articles/10.3389/neuro.06.004.2008/full)). Manifold projections (Isomap/UMAP/PHATE): cite algorithm originals (no canonical LLM-interpretability precedent).

## Ordinal / number / magnitude geometry
- **Singh, "Emergent Ordinal Geometry from Local Comparisons"** — [2606.01269](https://arxiv.org/abs/2606.01269). *linear/cyclic rank line.* Transformers trained on A<B,B<C form a rank line; transitive inference by geometry; topology linear-for-size, cyclic-for-months; a frozen LLM carries a reusable ordinal axis.
- **Bassi & Tomar, "Geometry of Ordinal Representations"** — [2607.04167](https://arxiv.org/abs/2607.04167). *curved place-cell 1D, degrading.* On Gemma-2/Qwen3, clean ordinal manifolds emerge for locally-computable ordinals; cross-position-integration tasks yield higher-dimensional / incoherent representations.
- **Gurnee et al. (Anthropic), "When Models Manipulate Manifolds: The Geometry of a Counting Task"** — [2601.04480](https://arxiv.org/abs/2601.04480). *curved place-cell manifold.* Counts on a curved manifold; attention heads twist it to a linear decision; causal.
- **Yuchi, Du & Eisner, "LLMs Know More About Numbers than They Can Say"** — [2602.07812](https://arxiv.org/abs/2602.07812). *log-linear + comparison subspace.* Relative order ("which is bigger") is encoded more robustly/separably than absolute magnitude.
- **Levy & Geva, "LMs Encode Numbers Using Digit Representations in Base 10"** — [2410.11781](https://arxiv.org/abs/2410.11781). *circular per-digit.* Numbers = independent per-digit mod-10 features (the per-digit landmine for any "magnitude axis").
- **Gould et al., "Successor Heads"** — [2312.09230](https://arxiv.org/abs/2312.09230). *modular mod-10.* Heads incrementing ordered tokens via shared modular features.
- **Heinzerling & Inui, "Monotonic Representation of Numeric Properties"** — [2403.10381](https://arxiv.org/abs/2403.10381). *linear monotone 1D.* A single steerable direction encodes numeric attributes monotonically.
- **Zhu, Dai & Sui, "LMs Encode the Value of Numbers Linearly"** — [2401.03735](https://arxiv.org/abs/2401.03735). *linear (log-space).* Linear probes read values; the "linear" pole of the log-vs-linear debate.
- **AlQuabeh et al., "Number Representations… Parallel to Human Perception"** — [2502.16147](https://arxiv.org/abs/2502.16147). *log-curved.* PCA reveals sublinear/log spacing local probes miss.
- **Cacioli, "Weber's Law in Transformer Magnitude Representations"** — [2603.20642](https://arxiv.org/abs/2603.20642). *log-curved.* Log geometry present even where a model discriminates at chance (geometry ≠ behavior).
- **Marjieh, Veselovsky, Griffiths & Sucholutsky, "What is a Number, That an LLM May Know It?"** — [2502.01540](https://arxiv.org/abs/2502.01540). *mixed (magnitude ⊗ string-edit).* Integer similarity entangles orthographic and numeric distance.
- **Hu, Niu & Varma, "The Representational Geometry of Number"** — [2602.06843](https://arxiv.org/abs/2602.06843). *linear, task-transferable.* Magnitude = separable linear direction; task subspaces linearly transformable.

## In-context construction of structure & manifold-vs-query-local
- **Park et al., "In-Context Learning of Representations"** — [2501.00070](https://arxiv.org/abs/2501.00070). *curved (grid/ring).* Context-driven reorganization to graph geometry; Dirichlet-energy alignment; pretrained semantics compete.
- **Kowalyshyn et al., "Belief or Circuitry? Causal Evidence for In-Context Graph Learning"** — [2605.08405](https://arxiv.org/abs/2605.08405). *nonlinear (orthogonal graph subspaces).* Persistent latent structure vs local copying, causally.
- **Hosseini et al., "Context Structure Reshapes the Representational Geometry of LMs"** — [2601.22364](https://arxiv.org/abs/2601.22364). *linear (straightening).* ICL straightens trajectories for continual prediction; inconsistent for structured tasks.
- **Xiong et al., "LLMs Reorganize Representational Geometry during ICL"** — [2605.28854](https://arxiv.org/abs/2605.28854). *geometry-reorg.*
- **"Provable Low-Frequency Bias of ICL of Representations"** — [2507.13540](https://arxiv.org/abs/2507.13540). *spectral.*
- **Dutta, Ansari & Das, "Limits of ICL beyond Functions using Partially Ordered Relation"** — [2506.13608](https://arxiv.org/abs/2506.13608). *behavioral.*
- **Hendel et al., "ICL Creates Task Vectors"** — [2310.15916](https://arxiv.org/abs/2310.15916); **Todd et al., "Function Vectors"** — [2310.15213](https://arxiv.org/abs/2310.15213). *method.* ICL compresses demos into an applied vector.
- **von Oswald et al., "Transformers Learn In-Context by Gradient Descent"** — [2212.07677](https://arxiv.org/abs/2212.07677); **Garg et al., "What Can Transformers Learn In-Context?"** — [2208.01066](https://arxiv.org/abs/2208.01066). *algorithmic.* ICL as implicit computation (the query-local alternative).
- **Wang et al., "Grokked Transformers are Implicit Reasoners"** — [2405.15071](https://arxiv.org/abs/2405.15071). *circuit.* Comparison generalizes OOD after grokking.
- **Yang et al., "Do LLMs Latently Perform Multi-Hop Reasoning?"** — [2402.16837](https://arxiv.org/abs/2402.16837). *latent path.*
- **Li et al., "Emergent World Representations (Othello-GPT)"** — [2210.13382](https://arxiv.org/abs/2210.13382) ↔ **Nanda, Lee & Wattenberg, "Emergent *Linear* Representations…"** — [2309.00941](https://arxiv.org/abs/2309.00941). *nonlinear→linear.* A "nonlinear" world model becomes linear under the right basis — re-basing can dissolve an apparent curve.
- **Karvonen, "Emergent World Models (Chess)"** — [2403.15498](https://arxiv.org/abs/2403.15498); **Spies et al., "Causal World Models in Maze-Solving"** — [2412.11867](https://arxiv.org/abs/2412.11867). *structured features, causal.*

## Spatial / grounded conceptual spaces
- **Tehenan, Moya, Long & Lin, "Linear Spatial World Models Emerge in LLMs"** — [2506.02996](https://arxiv.org/abs/2506.02996). *linear, causal, literal.* Antipodal opposites, orthogonal independents, additive composition, steering ~74%.
- **Gurnee & Tegmark, "LMs Represent Space and Time"** — [2310.02207](https://arxiv.org/abs/2310.02207). *linear, literal.* Linear geographic & temporal coordinates.
- **Jin et al., "More than Correlation: … Causal Representations of Space?"** — [2312.16257](https://arxiv.org/abs/2312.16257). *linear, causal.* Perturbing spatial reps degrades geospatial prediction.
- **Min et al., "Why Far Looks Up: Probing Spatial Representation in VLMs"** — [2605.30161](https://arxiv.org/abs/2605.30161). *mixed; axis entanglement.* Horizontal clean; vertical/depth entangled.
- **Patel & Pavlick, "Mapping LMs to Grounded Conceptual Spaces"** — [ICLR 2022](https://openreview.net/forum?id=gJcEM8sxHK). *linear map.* Few in-context anchors align a domain to a grounded grid.
- **Grand et al., "Semantic projection recovers rich human knowledge…"** — [Nature Hum. Behav. 2022](https://www.nature.com/articles/s41562-022-01316-8). *linear projection.* Antonym-axis projection recovers graded magnitude judgments.
- **Abdou et al., "Encode Perceptual Structure Without Grounding? (Color)"** — [2109.06129](https://arxiv.org/abs/2109.06129). *topological.*

## Cognitive effects / metaphor / abstraction transfer
- **Shaki, Kraus & Wooldridge, "Cognitive Effects in LLMs"** — [2308.14337](https://arxiv.org/abs/2308.14337). *behavioral.* GPT-3 reproduces priming, distance, SNARC, size-congruity — the one LLM link between magnitude and left-right (behavioral).
- **Binz & Schulz, "Using cognitive psychology to understand GPT-3"** — [2206.14576](https://arxiv.org/abs/2206.14576). *behavioral.*
- **Hagendorff, "Machine Psychology"** — [2303.13988](https://arxiv.org/abs/2303.13988). *framework.*
- **Sumita et al., "Cognitive Biases in LLMs: Survey"** — [2412.00323](https://arxiv.org/abs/2412.00323). *survey.*
- **Hu et al., "Metaphors are a Source of Cross-Domain Misalignment"** — [2601.03388](https://arxiv.org/abs/2601.03388). *feature-level, causal.*
- **Ghosh & Jiang, "Exploring Concreteness Through a Figurative Lens"** — [2604.18296](https://arxiv.org/abs/2604.18296). *1-D SVD subspace, causal steering.*

## Methods stack (steering / binding / entity substrate)
- Steering / causal subspace: **RepE** ([2310.01405](https://arxiv.org/abs/2310.01405)), **CAA** ([2312.06681](https://arxiv.org/abs/2312.06681)), **DAS** ([2303.02536](https://arxiv.org/abs/2303.02536)), **Path Patching** ([2304.05969](https://arxiv.org/abs/2304.05969)).
- Binding / entity substrate: **Feng & Steinhardt** ([2310.17191](https://arxiv.org/abs/2310.17191)), **Dai et al.** ([2409.05448](https://arxiv.org/abs/2409.05448)), **Gur-Arieh et al.** ([2510.06182](https://arxiv.org/abs/2510.06182), positional retrieval degrades mid-list — supports interior-only).
- Entity tracking: **Kim & Schuster** ([2305.02363](https://arxiv.org/abs/2305.02363)), **Prakash et al.** ([2402.14811](https://arxiv.org/abs/2402.14811)).
- Baseline: **Park, Choe & Veitch, "The Linear Representation Hypothesis…"** ([2311.03658](https://arxiv.org/abs/2311.03658)).

## Positioning
Prior work has separately established: (i) that novel in-context structure can
reorganize internal geometry (Park et al.; Kowalyshyn et al.); (ii) that ordinal
and magnitude quantities are often carried on *nonlinear* manifolds — helices,
circles, log-curves, place-cell tilings (Kantamneni & Tegmark; Engels et al.;
Levy & Geva; Gurnee et al.; Bassi & Tomar); and (iii) that a pretrained
magnitude/space direction exists and can be steered (Tehenan et al.; Heinzerling
& Inui; Hu et al.). Two recent results are especially close: Singh (2606.01269)
finds an emergent rank-line and a reusable ordinal axis, and Bassi & Tomar
(2607.04167) find that ordinal manifolds *degrade in the cross-position-integration
regime*. This project targets the intersection none of them covers: a
confound-clean, nonce-entity, **in-context-constructed** order, decoded
**interior-only** against a permutation null, characterized for **nonlinearity**
and localized in **depth** across **≥3 architectures**, with a causal test of
whether the novel order reuses a pretrained magnitude axis. The confound-clean
BCS design is, in particular, an adjudicator of exactly the integration regime
where ordinal-manifold coherence is reported to break down.
