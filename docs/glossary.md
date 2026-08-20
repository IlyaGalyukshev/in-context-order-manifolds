# Glossary — frozen terminology (M7)

Fixed terms for the paper, so the v1→v5 wobble doesn't leak into the text.

| Loose phrase used earlier | Frozen term | Meaning |
|---|---|---|
| "thin axis" / "weak manifold" | **low-variance ordinal subspace** | the coherent-order direction(s); small variance, supervised-readable, and (on hard) unsupervised-measurable |
| "local chaining" | **relational adjacency signal (induction-attributable)** | structure recoverable from adjacent-mention pairs without integration; the twin's decodable part; the ICLR-critique mechanism |
| "the map" (ambiguous) | **resting map** vs **evoked/assembled map** | representation at rest (roster/probe token before a question) vs assembled under a query (§6.15) |
| "coherence signal" | **coherence-specific increment** | real − incoherent-cycle-twin; what survives above the induction-attributable baseline |
| "manifold" (bare) | **in-context order geometry** | geometry of a context-specified (not pretrained) relational order |
| "hard vs easy" | **global-integration vs local-chaining regime** | random-regular (long-edge) vs banded (locally-chainable); the difficulty gate on the unsupervised coherence increment |
| "whitened RSA" | **crossnobis / whitened-cvRDM** | cross-validated Mahalanobis dissimilarity (Walther 2016); unbiased, noise-normalized |
| "declared vs derived" | **stated vs assembled order** | order given as a list (D2) vs inferred from shuffled binary relations (D1) |

**Estimand statement (say once, up front):** we measure a *single-prompt, single-read-per-entity* estimand — the most conservative in the field — not a centroid over exemplars/queries; centroids are shown only as a labeled caveat (the N40 thread).

**Standing methodological note:** the coherence-specific increment (crossnobis, Dirichlet-rank) is **difficulty-gated** — it is meaningful only in the global-integration regime; easy/locally-chainable stimuli let the twin absorb it. Every such number is reported on hard, with easy as the contrast.
