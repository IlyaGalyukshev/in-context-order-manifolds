#!/bin/bash
# ============================================================================
# X1 — full per-model sweep, CLUSTER-AGNOSTIC (mlc H100 or DGX V100).
# Called as:  run_scripts.sh <MODEL_ID>
#
# Comparability: datasets are generated deterministically from (config, seed) into a SHARED dir,
# so the SAME command produces IDENTICAL stimuli for every model → the whole gemma-4 ladder
# (E2B/E4B/12B on DGX, 31B on H100) is measured on exactly the same samples.
#
# Paths / knobs via env:
#   WORK_DIR    repo root (scripts/ + src/)         default /work
#   DATA_DIR    SHARED datasets dir                  default $WORK/data/sweep
#   OUT_ROOT    results root                         default $WORK/manifolds
#   MODEL_PATH  local weights dir (offline)          default /hf_models
#   DEVICE_MAP  transformers device_map              default auto
#   PER_CELL    stimuli per generation cell          default 40; also K, BATCH, GATE_LIMIT (see below)
#
# Smart sweep (vary ONE axis from the s0_zib/N12/hard/shuffle centre — full generator coverage,
# not a full cartesian): families(5) · N(7/9/12/16) · difficulty(easy/hard) · condition(shuffle/
# forward) · declared ladder(D1/D3/D4/D2) · hop-dial decay · structures(cyclic/grid2d/partial).
# Confound-clean core stays fixed: interior-only ranks, coherence-twin, all gates.
#
# Per-stage tolerant (a failing stage logs and the pipeline continues) + idempotent/resumable.
# ============================================================================
set -uo pipefail

MODEL_ID="${1:?usage: run_scripts.sh <MODEL_ID>}"
export MODEL_PATH="${MODEL_PATH:-/hf_models}"
WORK="${WORK_DIR:-/work}"; export PYTHONPATH="$WORK/src"; export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # anti-fragmentation over the long run (varying N)
TAG="$(echo "$MODEL_ID" | tr '/: ' '___')"
DATA="${DATA_DIR:-$WORK/data/sweep}"                 # SHARED across models (deterministic → identical)
OUT="${OUT_ROOT:-$WORK/manifolds}/$TAG"; A="$OUT/acts"
DEV="${DEVICE_MAP:-auto}"
PC="${PER_CELL:-40}"          # stimuli/cell — enough for RSA CIs + gate accuracy (was 60)
K="${K:-6}"                   # repeat-reads for crossnobis (6 is plenty; was 8)
BATCH="${BATCH:-24}"          # gate batch size — bump to use the GPUs (was 8)
GATE_LIMIT="${GATE_LIMIT:-150}"   # gate stimuli cap — enough for per-family accuracy (was 400)
mkdir -p "$OUT" "$DATA"
GEN=$WORK/scripts/generate_bcs.py; EXT=$WORK/scripts/extract_repeat.py; PRB=$WORK/scripts/probe_crossnobis.py
BAT=$WORK/scripts/run_battery.py; PAT=$WORK/scripts/patch_entity.py; STE=$WORK/scripts/steer_rank.py
FRM=$WORK/scripts/form_select.py
log(){ echo "[$(date '+%F %T')] $*"; }
step(){ log "════════════════════ $* ════════════════════"; }
run(){ log ">> $*"; "$@" && log "   ok" || log "   !! FAILED: $1 (stage continues)"; }
rsum(){ python3 - "$1" <<'PY' 2>/dev/null || true
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit()
for r in (d if isinstance(d,list) else [d]):
    if 'increment' not in r: continue
    tag=r.get('scheme','')
    for k in ('hop_reach','declared','pair_subset','probe_type','card_frac','difficulty'):
        if r.get(k) not in (None,''): tag+=f"/{k}={r[k]}"
    print("   ★ %-9s %-18s N%-2s incr=%+.3f (real=%.3f twin=%.3f) %s" % (
        str(r.get('family',''))[:9], tag[:18], r.get('n_items','?'),
        r['increment'], r.get('rsa_real',float('nan')), r.get('rsa_twin',float('nan')),
        'SIG' if r.get('sig_vs_twin') else 'ns'))
PY
}
probe(){ local j="$1"; shift; run python3 "$PRB" "$@" --json "$j"; rsum "$j"; }
MP=(--model "$TAG" --model-path "$MODEL_PATH" --role instruct)
PM=(--model "$TAG")
FAMS="s0_zib s0_quomp s1_size s1_loud s1_heat"

log "X1 sweep | model=$MODEL_ID | device_map=$DEV | per_cell=$PC | data=$DATA | out=$OUT"
python3 -c "import torch,transformers;print('[env] torch',torch.__version__,'| transformers',transformers.__version__,'| gpus',torch.cuda.device_count())"

# ---- 0. datasets (shared, deterministic) ----------------------------------
step "STAGE 0 — datasets (shared, deterministic → identical across models)"
[ -f "$DATA/core/stimuli.jsonl" ]   || run python3 "$GEN" --out "$DATA/core"   --families s0_zib,s0_quomp,s1_size,s1_loud,s1_heat --n-grid 7,9,12,16 --per-cell $PC --difficulty hard --conditions shuffle
[ -f "$DATA/ctrl/stimuli.jsonl" ]   || run python3 "$GEN" --out "$DATA/ctrl"   --families s0_zib --n-grid 12 --per-cell $PC --difficulty both --conditions shuffle,forward --declared list,adjacency,summary
[ -f "$DATA/hop/stimuli.jsonl" ]    || run python3 "$GEN" --out "$DATA/hop"    --families s0_zib,s1_size --n-grid 12,16 --per-cell $PC --difficulty hard --conditions shuffle --hopdial 1,2,3,4
[ -f "$DATA/struct/stimuli.jsonl" ] || run python3 "$GEN" --out "$DATA/struct" --families s0_zib,s1_size --n-grid 9,12 --per-cell $PC --difficulty hard --conditions shuffle --structures
log "core=$(wc -l <"$DATA/core/stimuli.jsonl" 2>/dev/null) ctrl=$(wc -l <"$DATA/ctrl/stimuli.jsonl" 2>/dev/null) hop=$(wc -l <"$DATA/hop/stimuli.jsonl" 2>/dev/null) struct=$(wc -l <"$DATA/struct/stimuli.jsonl" 2>/dev/null)"

# ---- 1. behaviour gate (core; capped subset — enough for the threshold) ----
step "STAGE 1 — behaviour gate"
run python3 "$BAT" "${MP[@]}" --device-map "$DEV" --batch-size "$BATCH" --sample-every 10 --limit "$GATE_LIMIT" \
  --stimuli "$DATA/core/stimuli.jsonl" --questions "$DATA/core/questions.jsonl" --out-dir "$OUT/battery"

# ---- 2. extraction (core + ctrl): card_mean / readout / neutral probe ----
step "STAGE 2 — extraction (k=$K)"
for DS in core ctrl; do for SRC in stimuli stimuli_null; do
  [ -f "$DATA/$DS/$SRC.jsonl" ] || continue
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci readout,card_mean --store rdm --stimuli "$DATA/$DS/$SRC.jsonl" --out "$A"
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe            --store rdm --stimuli "$DATA/$DS/$SRC.jsonl" --out "$A/probe"
done; done

# ---- 3. E1 manifold RSA: family axis (5 @ N12) + N-curve (s0_zib @ 7/9/12/16) ----
step "STAGE 3 — E1 manifold RSA (family axis + N-curve) — ★ = the manifold result"
for FAM in $FAMS; do for SC in card_mean readout; do
  probe "$OUT/rsa_${FAM}_${SC}_N12.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme $SC --n-items 12
done
  probe "$OUT/rsa_${FAM}_probe_N12.json" --acts "$A/probe" "${PM[@]}" --families $FAM --condition shuffle --scheme probe --n-items 12
done
for N in 7 9 12 16; do for SC in card_mean readout; do
  probe "$OUT/ncurve_${SC}_N${N}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme $SC --n-items $N
done; done

# ---- 4. difficulty gate (easy vs hard, s0_zib N12) ------------------------
step "STAGE 4 — difficulty gate (easy vs hard)"
for DIF in easy hard; do
  probe "$OUT/diff_${DIF}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme card_mean --n-items 12 --difficulty $DIF
done

# ---- 5. condition ceiling (shuffle=identification vs forward=ceiling) ------
step "STAGE 5 — condition (shuffle vs forward ceiling)"
for CON in shuffle forward; do
  probe "$OUT/cond_${CON}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition $CON --scheme card_mean --n-items 12 --difficulty hard
done

# ---- 6. E2 declared ladder (D1 none / D3 adjacency / D4 summary / D2 list) --
step "STAGE 6 — E2 declared ladder"
for DECL in none adjacency derived_summary list; do
  probe "$OUT/e2_${DECL}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme card_mean --declared $DECL --n-items 12
done

# ---- 7. R8 strata (stated 1-hop vs inferred multi-hop) --------------------
step "STAGE 7 — R8 strata (stated vs inferred)"
for FAM in s0_zib s1_size; do for SUB in onehop multihop; do
  probe "$OUT/r8_${FAM}_${SUB}.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme card_mean --n-items 12 \
    --pair-subset $SUB --stimuli "$DATA/core/stimuli.jsonl"
done; done

# ---- 8. decay curve (hop-dial derivation depth) ---------------------------
step "STAGE 8 — decay curve (hop-dial)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean --store rdm --stimuli "$DATA/hop/$SRC.jsonl" --out "$A/hop"
done
for RE in 1 2 3 4; do
  probe "$OUT/decay_r${RE}.json" --acts "$A/hop" "${PM[@]}" --families s0_zib,s1_size --condition shuffle --scheme card_mean --hop-reach $RE --n-items 16
done

# ---- 9. cross-form (structures: does geometry follow the latent?) ----------
step "STAGE 9 — cross-form (form-selection by structure)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean --store rdm+mean --stimuli "$DATA/struct/$SRC.jsonl" --out "$A/struct"
done
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families s0_zib,s0_quomp --condition shuffle --scheme card_mean --structure cyclic --templates line,ring,2block --json "$OUT/form_cyclic.json"
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families s0_zib,s0_quomp --condition shuffle --scheme card_mean --structure partial_order --templates line,2block --json "$OUT/form_partial.json"
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families "s1_size|s1_loud" --condition shuffle --scheme card_mean --structure grid2d --templates line,ring,2block,grid --json "$OUT/form_grid.json"

# ---- 10. causal: E9b entity-patch (CIK), steering, E7-Q / E8 ---------------
step "STAGE 10 — E9b entity-substitution patch (CIK toward-B vs toward-C)"
for SC in readout card_mean; do
  run python3 "$PAT" "${MP[@]}" --stimuli "$DATA/core/stimuli.jsonl" --families s0_zib --scheme $SC --n-stim 24 --n-pairs 3 --out "$OUT/e9b_patch_${SC}.parquet"
done
step "STAGE 10 — steering (graded axis-add + off-axis control)"
run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean,readout --store rdm+mean --stimuli "$DATA/core/stimuli.jsonl" --out "$OUT/acts_mean" --limit 60
run python3 "$STE" "${MP[@]}" --acts "$OUT/acts_mean" --stimuli "$DATA/core/stimuli.jsonl" --families s0_zib --scheme readout --n-stim 16 --alphas "-8,-4,-2,0,2,4,8" --n-offaxis 4 --out "$OUT/steer.parquet"
step "STAGE 10 — E7-Q (order/nonorder probe) + E8 (card-fraction dynamics)"
for PT in order nonorder; do
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe --probe-type $PT --store rdm --stimuli "$DATA/core/stimuli.jsonl" --out "$A/e7q_$PT"
  probe "$OUT/e7q_${PT}.json" --acts "$A/e7q_$PT" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe --probe-type $PT --n-items 12
done
run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe --card-fracs "0.25,0.5,0.75,1.0" --store rdm --stimuli "$DATA/core/stimuli.jsonl" --out "$A/e8"
for F in 0.25 0.5 0.75 1.0; do
  probe "$OUT/e8_frac${F}.json" --acts "$A/e8" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe --card-frac $F --n-items 12
done

step "X1 SWEEP DONE — model=$MODEL_ID"
log "results under $OUT:"; find "$OUT" -maxdepth 1 -type f | sed "s#$OUT/##" | sort
log "(hero figures: run scripts/make_figures.py locally on the pulled JSONs)"
