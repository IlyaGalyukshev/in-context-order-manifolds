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
FRM=$WORK/scripts/form_select.py; COUP=$WORK/scripts/probe_coupling.py
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

# ---- stage selection --------------------------------------------------------
# STAGES=all (default) | core | steer | comma list of:
#   gate,extract,probe,rsa,ncurve,diff,cond,e2,r8,decay,form,e9b,acts_mean,steer,coupling,e7q,e8
#   core  = gate,extract,rsa,acts_mean,coupling   (scale ladders: map + behaviour + coupling)
#   steer = gate,acts_mean,steer,coupling         (confirmatory steering on a fixed model set)
# CAUSAL_ONLY=1 (legacy) = e9b,acts_mean,steer,coupling (reuses existing acts_mean+battery)
STAGES="${STAGES:-all}"
[ "${CAUSAL_ONLY:-0}" = 1 ] && STAGES="e9b,acts_mean,steer,coupling"
case "$STAGES" in
  core)  STAGES="gate,extract,rsa,acts_mean,coupling" ;;
  steer) STAGES="gate,acts_mean,steer,coupling" ;;
esac
on(){ [ "$STAGES" = all ] || [[ ",$STAGES," == *",$1,"* ]]; }
EXTRACT_DS="${EXTRACT_DS:-core ctrl}"          # which datasets STAGE 2 extracts
ACTS_MEAN_LIMIT="${ACTS_MEAN_LIMIT:-60}"       # stimuli with stored mean activations (steering + coupling)
N_OFFAXIS="${N_OFFAXIS:-20}"                   # matched-norm off-axis controls (p-floor = 1/(n+1))
STEER_NSTIM="${STEER_NSTIM:-16}"; STEER_FAMS="${STEER_FAMS:-s0_zib,s1_size}"
STEER_SCHEME="${STEER_SCHEME:-readout}"; STEER_LAYERS="${STEER_LAYERS:-}"               # pre-registered steer layer(s); empty = 40/55/70% depth
log "stages=$STAGES | extract_ds=$EXTRACT_DS | acts_mean_limit=$ACTS_MEAN_LIMIT | n_offaxis=$N_OFFAXIS | steer_layers=${STEER_LAYERS:-default}"

# ---- 0. datasets (shared, deterministic) ----------------------------------
step "STAGE 0 — datasets (shared, deterministic → identical across models)"
[ -f "$DATA/core/stimuli.jsonl" ]   || run python3 "$GEN" --out "$DATA/core"   --families s0_zib,s0_quomp,s1_size,s1_loud,s1_heat --n-grid 7,9,12,16 --per-cell $PC --difficulty hard --conditions shuffle
[ -f "$DATA/ctrl/stimuli.jsonl" ]   || run python3 "$GEN" --out "$DATA/ctrl"   --families s0_zib --n-grid 12 --per-cell $PC --difficulty both --conditions shuffle,forward --declared list,adjacency,summary
[ -f "$DATA/hop/stimuli.jsonl" ]    || run python3 "$GEN" --out "$DATA/hop"    --families s0_zib,s1_size --n-grid 12,16 --per-cell $PC --difficulty hard --conditions shuffle --hopdial 1,2,3,4
[ -f "$DATA/struct/stimuli.jsonl" ] || run python3 "$GEN" --out "$DATA/struct" --families s0_zib,s1_size --n-grid 9,12 --per-cell $PC --difficulty hard --conditions shuffle --structures
log "core=$(wc -l <"$DATA/core/stimuli.jsonl" 2>/dev/null) ctrl=$(wc -l <"$DATA/ctrl/stimuli.jsonl" 2>/dev/null) hop=$(wc -l <"$DATA/hop/stimuli.jsonl" 2>/dev/null) struct=$(wc -l <"$DATA/struct/stimuli.jsonl" 2>/dev/null)"

# ---- 1. behaviour gate (core; capped subset — enough for the threshold) ----
if on gate; then step "STAGE 1 — behaviour gate"
run python3 "$BAT" "${MP[@]}" --device-map "$DEV" --batch-size "$BATCH" --sample-every 10 --limit "$GATE_LIMIT" \
  --stimuli "$DATA/core/stimuli.jsonl" --questions "$DATA/core/questions.jsonl" --out-dir "$OUT/battery"
fi

# ---- 2. extraction: card_mean / readout (+ neutral probe locus) ------------
if on extract; then step "STAGE 2 — extraction (k=$K, datasets: $EXTRACT_DS)"
for DS in $EXTRACT_DS; do for SRC in stimuli stimuli_null; do
  [ -f "$DATA/$DS/$SRC.jsonl" ] || continue
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci readout,card_mean --store rdm --stimuli "$DATA/$DS/$SRC.jsonl" --out "$A"
  on probe && run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe --store rdm --stimuli "$DATA/$DS/$SRC.jsonl" --out "$A/probe"
done; done
fi

# ---- 3. E1 manifold RSA: family axis (5 @ N12) + N-curve (s0_zib @ 7/9/12/16) ----
if on rsa; then step "STAGE 3 — E1 manifold RSA (family axis) — ★ = the manifold result"
for FAM in $FAMS; do for SC in card_mean readout; do
  probe "$OUT/rsa_${FAM}_${SC}_N12.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme $SC --n-items 12
done
  on probe && probe "$OUT/rsa_${FAM}_probe_N12.json" --acts "$A/probe" "${PM[@]}" --families $FAM --condition shuffle --scheme probe --n-items 12
done
fi
if on ncurve; then step "STAGE 3b — N-curve"
for N in 7 9 12 16; do for SC in card_mean readout; do
  probe "$OUT/ncurve_${SC}_N${N}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme $SC --n-items $N
done; done
fi

# ---- 4. difficulty gate (easy vs hard, s0_zib N12) ------------------------
if on diff; then step "STAGE 4 — difficulty gate (easy vs hard)"
for DIF in easy hard; do
  probe "$OUT/diff_${DIF}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme card_mean --n-items 12 --difficulty $DIF
done
fi

# ---- 5. condition ceiling (shuffle=identification vs forward=ceiling) ------
if on cond; then step "STAGE 5 — condition (shuffle vs forward ceiling)"
for CON in shuffle forward; do
  probe "$OUT/cond_${CON}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition $CON --scheme card_mean --n-items 12 --difficulty hard
done
fi

# ---- 6. E2 declared ladder (D1 none / D3 adjacency / D4 summary / D2 list) --
if on e2; then step "STAGE 6 — E2 declared ladder"
for DECL in none adjacency derived_summary list; do
  probe "$OUT/e2_${DECL}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme card_mean --declared $DECL --n-items 12
done
fi

# ---- 7. R8 strata (stated 1-hop vs inferred multi-hop) --------------------
if on r8; then step "STAGE 7 — R8 strata (stated vs inferred)"
for FAM in s0_zib s1_size; do for SUB in onehop multihop; do
  probe "$OUT/r8_${FAM}_${SUB}.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme card_mean --n-items 12 \
    --pair-subset $SUB --stimuli "$DATA/core/stimuli.jsonl"
done; done
fi

# ---- 8. decay curve (hop-dial derivation depth) ---------------------------
if on decay; then step "STAGE 8 — decay curve (hop-dial)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean --store rdm --stimuli "$DATA/hop/$SRC.jsonl" --out "$A/hop"
done
for RE in 1 2 3 4; do
  probe "$OUT/decay_r${RE}.json" --acts "$A/hop" "${PM[@]}" --families s0_zib,s1_size --condition shuffle --scheme card_mean --hop-reach $RE --n-items 16
done
fi

# ---- 9. cross-form (structures: does geometry follow the latent?) ----------
if on form; then step "STAGE 9 — cross-form (form-selection by structure)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean --store rdm+mean --stimuli "$DATA/struct/$SRC.jsonl" --out "$A/struct"
done
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families s0_zib,s0_quomp --condition shuffle --scheme card_mean --structure cyclic --templates line,ring,2block --json "$OUT/form_cyclic.json"
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families s0_zib,s0_quomp --condition shuffle --scheme card_mean --structure partial_order --templates line,2block --json "$OUT/form_partial.json"
run python3 "$FRM" --acts "$A/struct" "${PM[@]}" --families "s1_size|s1_loud" --condition shuffle --scheme card_mean --structure grid2d --templates line,ring,2block,grid --json "$OUT/form_grid.json"
fi

# ---- 10. causal: E9b entity-patch, mean activations, steering --------------
if on e9b; then step "STAGE 10 — E9b entity-substitution patch (CIK toward-B vs toward-C)"
for SC in readout card_mean; do
  run python3 "$PAT" "${MP[@]}" --stimuli "$DATA/core/stimuli.jsonl" --families s0_zib --scheme $SC --n-stim 24 --n-pairs 3 --out "$OUT/e9b_patch_${SC}.parquet"
done
fi
if on acts_mean; then step "STAGE 10a — mean activations for steering/coupling (limit $ACTS_MEAN_LIMIT)"
run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --loci card_mean,readout --store rdm+mean --stimuli "$DATA/core/stimuli.jsonl" --out "$OUT/acts_mean" --limit "$ACTS_MEAN_LIMIT"
fi
if on steer; then step "STAGE 10b — steering (graded axis-add vs $N_OFFAXIS matched-norm off-axis controls)"
run python3 "$STE" "${MP[@]}" --acts "$OUT/acts_mean" --stimuli "$DATA/core/stimuli.jsonl" --families "$STEER_FAMS" --scheme "$STEER_SCHEME" \
  --n-stim "$STEER_NSTIM" --alphas="-8,-4,-2,0,2,4,8" --n-offaxis "$N_OFFAXIS" ${STEER_LAYERS:+--steer-layers "$STEER_LAYERS"} --out "$OUT/steer.parquet"
fi

# ---- 11. P0.2 coupling: resting geometry → answer correctness (CPU) → canonical coupling/ -----
if on coupling; then step "STAGE 11 — P0.2 coupling (resting margin predicts pair correctness) → coupling/<family>.json"
mkdir -p "$OUT/coupling"
for FAM in $FAMS; do
  run python3 "$COUP" --model "$TAG" --acts "$OUT/acts_mean" --battery "$OUT/battery/battery_$TAG.jsonl" \
    --questions "$DATA/core/questions.jsonl" --stimuli "$DATA/core/stimuli.jsonl" --families "$FAM" --q-family pairwise --n-boot 500 \
    --json "$OUT/coupling/$FAM.json"
done
fi

# ---- 12. E7-Q (order/nonorder probe) + E8 (card-fraction dynamics) ----------
if on e7q; then step "STAGE 12 — E7-Q (order / non-order probe)"
for PT in ${E7Q_TYPES:-neutral order nonorder}; do
  for SRC in stimuli stimuli_null; do   # twin too → probe reports real − twin
    run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe --probe-type $PT --store rdm --stimuli "$DATA/core/$SRC.jsonl" --out "$A/e7q_$PT"
  done
  probe "$OUT/e7q_${PT}.json" --acts "$A/e7q_$PT" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe --probe-type $PT --n-items 12
done
fi
if on e8; then step "STAGE 13 — E8 (card-fraction dynamics)"
for SRC in stimuli stimuli_null; do   # twin too → probe reports real − twin
  run python3 "$EXT" "${MP[@]}" --device-map "$DEV" --k "$K" --probe --card-fracs "0.25,0.5,0.75,1.0" --store rdm --stimuli "$DATA/core/$SRC.jsonl" --out "$A/e8"
done
for F in 0.25 0.5 0.75 1.0; do
  probe "$OUT/e8_frac${F}.json" --acts "$A/e8" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe --card-frac $F --n-items 12
done
fi

# ---- optional: auto-sync results to the canonical HF dataset (online hosts, e.g. DGX) --------
if [ "${SYNC_HF:-0}" = 1 ]; then
  step "SYNC — push $OUT to HF dataset (canonical results/<model>/)"
  run python3 "$WORK/scripts/sync_to_hf.py" "$OUT" ${SYNC_ACTS:+--acts}
fi

step "RESULT SUMMARY — numbers printed to the log (closed contour: read via 'mlc job logs <job>', no file-dragging)"
python3 "$WORK/scripts/summarize.py" "$OUT" || true
step "DONE — model=$MODEL_ID stages=$STAGES"
