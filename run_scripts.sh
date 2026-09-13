#!/bin/bash
# ============================================================================
# X1 — full frozen per-model pipeline on ONE model (2×H100).
# Called by .ml-job-preset.yml as:  run_scripts.sh <MODEL_ID>
# Weights served locally at $MODEL_PATH; results persist in the `manifolds` storage.
#
# Frozen order (per the X1 plan):
#   0 datasets  1 gate  2 E1-crossnobis(k=8, card_mean/readout/probe)  3 E2 ladder(D1/D3/D4/D2/D2-long)
#   4 R8 strata(stated vs inferred)  5 decay curve(hop-dial)  6 E9b entity-patch(CIK)
#   7 steering  8 E7-Q / E8  (hero figures are done locally from the JSONs).
#
# NOT `set -e`: each stage is tolerant — a failing stage is logged and the pipeline continues,
# so partial results always land. Idempotent: re-submitting the job resumes (extraction skips
# existing npz, the battery resumes), which is how a run longer than one 48h slot completes.
# ============================================================================
set -uo pipefail

MODEL_ID="${1:?usage: run_scripts.sh <MODEL_ID>}"
export MODEL_PATH="${MODEL_PATH:-/hf_models}"
export PYTHONPATH=/work/src TOKENIZERS_PARALLELISM=false
TAG="$(echo "$MODEL_ID" | tr '/: ' '___')"
S=/work/manifolds/$TAG          # results (persistent storage)
D=/work/data                    # datasets (ephemeral, regenerated deterministically)
A=$S/acts                       # crossnobis RDM acts
mkdir -p "$S" "$D"
log(){ echo "[$(date '+%F %T')] $*"; }
step(){ log "════════════════════ $* ════════════════════"; }
run(){ log ">> $*"; "$@" && log "   ok" || log "   !! FAILED: $1 (stage continues)"; }
# print a clean ★ headline (incr / real / twin / SIG) from a crossnobis result json
rsum(){ python3 - "$1" <<'PY' 2>/dev/null || true
import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: sys.exit()
for r in (d if isinstance(d,list) else [d]):
    if 'increment' not in r: continue
    tag=r.get('scheme','')
    for k in ('hop_reach','declared','pair_subset','probe_type','card_frac'):
        if r.get(k) not in (None,''): tag+=f"/{k}={r[k]}"
    print("   ★ %-8s %-16s N%-2s incr=%+.3f (real=%.3f twin=%.3f) %s" % (
        r.get('family','')[:8], tag[:16], r.get('n_items','?'),
        r['increment'], r.get('rsa_real',float('nan')), r.get('rsa_twin',float('nan')),
        'SIG' if r.get('sig_vs_twin') else 'ns'))
PY
}
# run a crossnobis probe then immediately show its ★ result:  probe <jsonpath> <probe args...>
probe(){ local j="$1"; shift; run python3 "$PRB" "$@" --json "$j"; rsum "$j"; }

MP=(--model "$TAG" --model-path "$MODEL_PATH" --role instruct)   # our scripts' cluster model args
PM=(--model "$TAG")                                              # probe (no model load)
GEN=/work/scripts/generate_bcs.py; EXT=/work/scripts/extract_repeat.py
PRB=/work/scripts/probe_crossnobis.py; BAT=/work/scripts/run_battery.py
PAT=/work/scripts/patch_entity.py; STE=/work/scripts/steer_rank.py

log "X1 pipeline | model=$MODEL_ID | results -> $S"
python3 -c "import torch,transformers;print('[env] torch',torch.__version__,'| transformers',transformers.__version__,'| gpus',torch.cuda.device_count())"

# ---- 0. datasets ----------------------------------------------------------
step "STAGE 0 — datasets"
[ -f "$D/main/stimuli.jsonl" ] || run python3 "$GEN" --out "$D/main" \
  --families s0_zib,s1_loud,s1_size --n-grid 9,12 --per-cell 120 --difficulty hard --conditions shuffle \
  --declared list,adjacency,summary --declared-pad 24
[ -f "$D/hop/stimuli.jsonl" ] || run python3 "$GEN" --out "$D/hop" \
  --families s0_zib,s1_size --n-grid 12,16 --per-cell 100 --difficulty hard --conditions shuffle --hopdial 1,2,3,4
log "main=$(wc -l <"$D/main/stimuli.jsonl" 2>/dev/null) hop=$(wc -l <"$D/hop/stimuli.jsonl" 2>/dev/null)"

# ---- 1. behaviour gate ----------------------------------------------------
step "STAGE 1 — behaviour gate (pairwise + reconstruction)"
run python3 "$BAT" "${MP[@]}" --device-map auto --batch-size 8 --sample-every 10 \
  --stimuli "$D/main/stimuli.jsonl" --questions "$D/main/questions.jsonl" --out-dir "$S/battery"

# ---- 2. E1-hard crossnobis (k=8; card_mean + readout + neutral probe) ------
step "STAGE 2 — E1 extraction (k=8: card_mean/readout + neutral probe)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --loci readout,card_mean --store rdm \
    --stimuli "$D/main/$SRC.jsonl" --out "$A"
  run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --probe --store rdm \
    --stimuli "$D/main/$SRC.jsonl" --out "$A/probe"
done
step "STAGE 2 — E1 interior RSA (real vs coherence-twin) — ★ = the manifold result"
for FAM in s0_zib s1_loud s1_size; do for SC in card_mean readout; do for N in 9 12; do
  probe "$S/rsa_${FAM}_${SC}_N${N}.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme $SC --n-items $N
done; done; done
for FAM in s0_zib s1_loud s1_size; do for N in 9 12; do
  probe "$S/rsa_${FAM}_probe_N${N}.json" --acts "$A/probe" "${PM[@]}" --families $FAM --condition shuffle --scheme probe --n-items $N
done; done

# ---- 3. E2 declared ladder (D1 / D3-adjacency / D4-summary / D2-list / D2-long) ----
step "STAGE 3 — E2 declared ladder (RSA per declared mode, from the main acts)"
# none=D1 derived · adjacency=D3 · derived_summary=D4 · list=D2 (+D2-long padded variant folded in)
for DECL in none adjacency derived_summary list; do for N in 9 12; do
  probe "$S/e2_${DECL}_N${N}.json" --acts "$A" "${PM[@]}" --families s0_zib --condition shuffle --scheme card_mean \
    --declared $DECL --n-items $N
done; done

# ---- 4. R8 strata (stated 1-hop vs inferred multi-hop) --------------------
step "STAGE 4 — R8 strata (stated vs inferred)"
for FAM in s0_zib s1_loud s1_size; do for SUB in onehop multihop; do
  probe "$S/r8_${FAM}_${SUB}.json" --acts "$A" "${PM[@]}" --families $FAM --condition shuffle --scheme card_mean --n-items 12 \
    --pair-subset $SUB --stimuli "$D/main/stimuli.jsonl"
done; done

# ---- 5. decay curve (hop-dial: derivation depth) --------------------------
step "STAGE 5 — decay curve extraction (hop-dial)"
for SRC in stimuli stimuli_null; do
  run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --loci card_mean --store rdm \
    --stimuli "$D/hop/$SRC.jsonl" --out "$A/hop"
done
step "STAGE 5 — decay curve RSA (per reach)"
for RE in 1 2 3 4; do
  probe "$S/decay_r${RE}.json" --acts "$A/hop" "${PM[@]}" --families s0_zib,s1_size --condition shuffle --scheme card_mean \
    --hop-reach $RE --n-items 16
done

# ---- 6. E9b entity-substitution patch (CIK) ------------------------------
step "STAGE 6 — E9b entity-substitution patch (CIK toward-B vs toward-C control)"
for SC in readout card_mean; do
  run python3 "$PAT" "${MP[@]}" --stimuli "$D/main/stimuli.jsonl" --families s0_zib --scheme $SC \
    --n-stim 24 --n-pairs 3 --out "$S/e9b_patch_${SC}.parquet"
done

# ---- 7. steering (E9a graded axis-add) ------------------------------------
step "STAGE 7 — steering (graded axis-add + off-axis control)"
run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --loci card_mean,readout --store rdm+mean \
  --stimuli "$D/main/stimuli.jsonl" --out "$S/acts_mean" --limit 60
run python3 "$STE" "${MP[@]}" --acts "$S/acts_mean" --stimuli "$D/main/stimuli.jsonl" --families s0_zib \
  --scheme readout --n-stim 16 --alphas "-8,-4,-2,0,2,4,8" --n-offaxis 4 --out "$S/steer.parquet"

# ---- 8. E7-Q assembly ladder + E8 dynamics -------------------------------
step "STAGE 8 — E7-Q (order/nonorder probe) + E8 (card-fraction dynamics)"
for PT in order nonorder; do
  run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --probe --probe-type $PT --store rdm \
    --stimuli "$D/main/stimuli.jsonl" --out "$A/e7q_$PT"
  probe "$S/e7q_${PT}.json" --acts "$A/e7q_$PT" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe \
    --probe-type $PT --n-items 12
done
run python3 "$EXT" "${MP[@]}" --device-map auto --k 8 --probe --card-fracs "0.25,0.5,0.75,1.0" --store rdm \
  --stimuli "$D/main/stimuli.jsonl" --out "$A/e8"
for F in 0.25 0.5 0.75 1.0; do
  probe "$S/e8_frac${F}.json" --acts "$A/e8" "${PM[@]}" --families s0_zib --condition shuffle --scheme probe \
    --card-frac $F --n-items 12
done

step "X1 DONE — model=$MODEL_ID"
log "result JSONs/parquets under $S:"; find "$S" -maxdepth 1 -type f | sed "s#$S/##" | sort
log "(hero figures: run scripts/make_figures.py locally on the pulled $S JSONs)"
