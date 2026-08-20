# Pre-registration — confirmatory wave E1–E4 (v5 program)

Frozen decision rules for the P1 experiments, per the plan's M6(д)/§11.3. Written to defend against
the garden-of-forking-paths given the project's v1→v4.9 history. Load-bearing claims require **≥3
models and a bootstrap CI over stimuli**; the held-out layer is the MAIN number, argmax an optimistic
bound; every headline carries n + CI. Interior-only (ranks 3..N−2) is primary.

**Standing control discovered 2026-08-20 (must be honored):** the crossnobis / Dirichlet real−twin
increment is **difficulty-gated**. On *easy* (banded, locally-chainable) stimuli the incoherent-cycle
twin chains locally and captures the structure → real ≈ twin *trivially*. The coherence increment is
only meaningful on *hard* (random-regular, long-edge, global-integration-required) stimuli. **Every
E1–E4 crossnobis/Dirichlet cell is run on difficulty=hard; easy is reported only as the contrast that
shows the twin absorbing local structure.** (E1's first easy-only run was retracted for this reason.)

---

## E1 — repeat-read + crossnobis: is the resting coherence geometry readable WITHOUT supervision?
- **Design:** {qwen3-4b, olmo3-7b, gemma-4-12b} × {s0_zib(+s1_loud)} × N∈{9,12} × {real, twin} ×
  k=8 repeat-reads × loci {card_mean, last_token, readout}. **difficulty=hard.** Metrics: whitened
  crossnobis RSA to the line ideal (M2), Dirichlet smoothness gap (M5).
- **H1:** whitened-RSA(real) − whitened-RSA(twin) CI>0 on card_mean, hard.
- **Decision:** CI>0 in ≥2/3 models at N12 (hard) → resting map carries coherence geometry readable
  unsupervised (difficulty-gated). If ns → coherence is supervised-only (query-local narrative).
  Secondary: in-card ≫ readout must hold in the whitened metric regardless (method check).

## E2 — declared × derived: is a STATED order stored, a DERIVED order assembled?
- **Design:** D1 derived (current) vs D2 declared-list, content-matched (same entities/order), hard.
  **Locus must be comparable** — the naive D2 "card" (a bare list item) ≠ D1's relational card;
  compare at the READOUT (roster, identical in both) or a matched probe locus, not card_mean.
- **H2:** whitened-RSA(declared) ≫ whitened-RSA(derived) at the comparable locus.
- **Decision:** declared − derived CI>0 → "declared stored, derived assembled". Either sign is
  publishable. (First D2 run 2026-08-20 was locus-confounded → redesign pending.)

## E3 — redundancy r × {verbatim, paraphrase}
- **Design:** r∈{1,2,4,8} card repeats (uniform → mention-freq ⟂ rank preserved); verbatim vs
  paraphrase surface. real+twin, hard, {gemma, qwen}, N12.
- **H3:** verbatim-r raises twin-decode ≈ real (induction path); paraphrase-r raises the gap
  (abstraction). gap(r) nonlinear/threshold (Park emergence).
- **Decision:** paraphrase>verbatim on gap-slope (CI on slope) → induction vs integration separated
  without head surgery. Length-vs-graph control: pad r=1 to r=8 token length → collapse tracks graph.

## E4 — determinacy dial m
- **Design:** m∈{1,2,4,8} independent chain fragments over fixed N (m=1 full order; m→N = twin limit),
  via the partial-order builder. q(m)=resolvable-pair fraction. real, hard, {gemma, qwen}, N12.
- **H4:** crossnobis gap and behavior both decline monotonically with q(m); behavior on resolvable
  pairs stays stable → "map global, answers local".
- **Decision:** monotone gap(q) with CI → dose-response replaces the binary real/twin.

---

*Status log:* E1 run (easy retracted → hard confirmation `e1hard_20260820` running); E2 D2 built,
locus-confounded; E3/E4 generators pending. Results → report_2026-08-10 §6.24.
