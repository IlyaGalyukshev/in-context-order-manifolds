#!/bin/bash
# ============================================================================
# Cluster entry for the in-context ORDER-MANIFOLD experiments (2×H100 job).
# Invoked by .ml-job-preset.yml as:  run_scripts.sh <MODEL_ID>
# The model weights are served LOCALLY at $MODEL_PATH (HF offline); the whole
# repo is uploaded to /work. Results + logs land under $OUT_ROOT so you can
# tail the job to monitor every stage.
#
# Knobs (env):
#   MODEL_PATH   local weights dir (default /mr_models, set by the .yml)
#   ROLE         instruct | base            (chat-template vs raw)   default instruct
#   EXPERIMENT   core | hopdial             default core
#                  core    = scale-insurance: interior order-manifold (real vs
#                            coherence-twin) + behaviour gate, N∈{9,12,16}
#                  hopdial = derivation-depth dose-response (reach 1..4)
#   LIMIT        cap #stimuli per extract pass (smoke); empty = full
# ============================================================================
set -uo pipefail

MODEL_ID="${1:-${MODEL_ID:-unknown}}"
MODEL_PATH="${MODEL_PATH:-/mr_models}"
ROLE="${ROLE:-instruct}"
EXPERIMENT="${EXPERIMENT:-core}"
WORK="${WORK_DIR:-/work}"
OUT_ROOT="${OUT_ROOT:-$WORK/results}"
TAG="$(echo "$MODEL_ID" | sed 's#[/: ]#__#g')"
OUT="$OUT_ROOT/$TAG/$EXPERIMENT"
DATA="$WORK/data/manifold_${EXPERIMENT}"
mkdir -p "$OUT" "$DATA"
export PYTHONPATH="$WORK/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

log(){ echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
banner(){ log "############################################################"; log "# $*"; log "############################################################"; }
LIM=(); [ -n "${LIMIT:-}" ] && LIM=(--limit "$LIMIT")

banner "ORDER-MANIFOLD | model=$MODEL_ID | experiment=$EXPERIMENT | role=$ROLE"
log "MODEL_PATH=$MODEL_PATH   OUT=$OUT   DATA=$DATA   LIMIT=${LIMIT:-<none>}"
python3 -c "import torch,transformers as t; print(f'[env] torch {torch.__version__} | transformers {t.__version__} | cuda {torch.cuda.is_available()} | n_gpu {torch.cuda.device_count()}')" || true

# ---- Stage 0: dataset (deterministic — generate once, reused on resume) -----
banner "STAGE 0/3 — dataset"
if [ ! -f "$DATA/stimuli.jsonl" ]; then
  if [ "$EXPERIMENT" = "hopdial" ]; then
    python3 "$WORK/scripts/generate_bcs.py" --out "$DATA" --families s0_zib,s1_size \
      --n-grid 9,12,16 --per-cell 100 --difficulty hard --conditions shuffle --hopdial 1,2,3,4
  else
    python3 "$WORK/scripts/generate_bcs.py" --out "$DATA" --families s0_zib,s1_size \
      --n-grid 9,12,16 --per-cell 100 --difficulty both --conditions shuffle
  fi
else
  log "dataset present — reuse"
fi
log "stimuli=$(wc -l <"$DATA/stimuli.jsonl")  twin=$(wc -l <"$DATA/stimuli_null.jsonl")  questions=$(wc -l <"$DATA/questions.jsonl")"

# ---- Stage 1: extraction (real + coherence-twin), sharded over both H100s ---
banner "STAGE 1/3 — extraction (readout + card_mean, k=6)"
COMMON=(--model "$TAG" --model-path "$MODEL_PATH" --role "$ROLE" --device-map auto \
        --k 6 --loci readout,card_mean --store rdm+mean --out "$OUT/acts" "${LIM[@]}")
log ">> real stimuli";  python3 "$WORK/scripts/extract_repeat.py" "${COMMON[@]}" --stimuli "$DATA/stimuli.jsonl"      || { log "!! EXTRACT real FAILED";  exit 1; }
log ">> twin stimuli";  python3 "$WORK/scripts/extract_repeat.py" "${COMMON[@]}" --stimuli "$DATA/stimuli_null.jsonl" || { log "!! EXTRACT twin FAILED";  exit 1; }
log "extracted npz: $(ls "$OUT/acts/$TAG"/*.npz 2>/dev/null | wc -l)"

# ---- Stage 2: geometry — interior RSA(real) − RSA(coherence-twin) -----------
banner "STAGE 2/3 — interior order-manifold RSA (real vs coherence-twin)"
for SC in card_mean readout; do for N in 9 12 16; do
  log "-- scheme=$SC N=$N --"
  python3 "$WORK/scripts/probe_crossnobis.py" --acts "$OUT/acts" --model "$TAG" \
     --families s0_zib,s1_size --condition shuffle --scheme "$SC" --n-items "$N" \
     --n-boot 400 --n-perm 150 --json "$OUT/rsa_${SC}_N${N}.json" 2>&1 | grep -iE "rsa_real|incr|SIG|peak|ERROR" || true
done; done
if [ "$EXPERIMENT" = "hopdial" ]; then
  log "-- hop-dial decay curve (card_mean, N16) --"
  for RE in 1 2 3 4; do
    python3 "$WORK/scripts/probe_crossnobis.py" --acts "$OUT/acts" --model "$TAG" \
       --families s0_zib,s1_size --condition shuffle --scheme card_mean --hop-reach "$RE" --n-items 16 \
       --n-boot 400 --n-perm 150 --json "$OUT/rsa_hop_r${RE}.json" 2>&1 | grep -iE "incr|SIG|ERROR" || true
  done
fi

# ---- Stage 3: behaviour gate (integration-requiring questions) --------------
banner "STAGE 3/3 — behaviour gate"
python3 "$WORK/scripts/run_battery.py" --model "$TAG" --model-path "$MODEL_PATH" --role "$ROLE" \
   --device-map auto --stimuli "$DATA/stimuli.jsonl" --questions "$DATA/questions.jsonl" \
   --out-dir "$OUT/battery" "${LIM[@]}" 2>&1 | tail -25 || log "!! battery returned nonzero"

banner "DONE — model=$MODEL_ID experiment=$EXPERIMENT"
log "results tree:"; find "$OUT" -maxdepth 2 -type f | sed "s#$OUT/##" | sort
