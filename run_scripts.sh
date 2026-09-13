#!/bin/bash
# In-context ORDER-MANIFOLD — full per-model paper pipeline for ONE model.
# Called by .ml-job-preset.yml as:  run_scripts.sh <MODEL_ID>
# Weights are served locally at $MODEL_PATH; results persist in the `manifolds` storage.
set -e

MODEL_ID="${1:?usage: run_scripts.sh <MODEL_ID>}"
export MODEL_PATH="${MODEL_PATH:-/hf_models}"
export PYTHONPATH=/work/src

TAG="$(echo "$MODEL_ID" | tr '/: ' '___')"
DATA=/work/data
ACTS=/work/manifolds/acts
RES=/work/manifolds/$TAG
mkdir -p "$RES"
log(){ echo "[$(date '+%F %T')] $*"; }

log "MODEL=$MODEL_ID  MODEL_PATH=$MODEL_PATH  ->results $RES"
python3 -c "import torch,transformers;print('[env] torch',torch.__version__,'| transformers',transformers.__version__,'| gpus',torch.cuda.device_count())"

# 1. dataset (deterministic — generated once, reused on resume) ---------------
log "STEP 1/4  dataset"
[ -f "$DATA/stimuli.jsonl" ] || python3 /work/scripts/generate_bcs.py --out "$DATA" \
  --families s0_zib,s1_size --n-grid 9,12,16 --per-cell 100 --difficulty both --conditions shuffle
log "  stimuli=$(wc -l <"$DATA/stimuli.jsonl") twin=$(wc -l <"$DATA/stimuli_null.jsonl") questions=$(wc -l <"$DATA/questions.jsonl")"

# 2. behaviour gate (integration-requiring questions) ------------------------
log "STEP 2/4  behaviour gate"
python3 /work/scripts/run_battery.py --model "$TAG" --model-path "$MODEL_PATH" --device-map auto \
  --batch-size 8 --stimuli "$DATA/stimuli.jsonl" --questions "$DATA/questions.jsonl" --out-dir "$RES/battery"

# 3. extraction (real + coherence-twin), sharded over both H100s -------------
log "STEP 3/4  extraction (readout + card_mean, k=6)"
for SRC in stimuli stimuli_null; do
  python3 /work/scripts/extract_repeat.py --model "$TAG" --model-path "$MODEL_PATH" --device-map auto \
    --stimuli "$DATA/$SRC.jsonl" --out "$ACTS" --k 6 --loci readout,card_mean --store rdm
done

# 4. interior order-manifold RSA (real vs coherence-twin), per locus × N ------
log "STEP 4/4  interior RSA (real vs coherence-twin)"
for SC in card_mean readout; do for N in 9 12 16; do
  python3 /work/scripts/probe_crossnobis.py --acts "$ACTS" --model "$TAG" --families s0_zib,s1_size \
    --condition shuffle --scheme "$SC" --n-items "$N" --json "$RES/rsa_${SC}_N${N}.json"
done; done

log "DONE  results in $RES"
ls -R "$RES"
