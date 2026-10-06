#!/bin/bash
# Developmental sweep over OLMo-3-7B training checkpoints (base model, open stage1/2/3 revisions):
# does the in-context order map at the entity token appear before the model can answer?
# One checkpoint on disk at a time (download -> gate + readout extraction -> per_stim_rsa -> delete).
#
#   CKPTS="stage1-step10000 stage1-step100000 ..." WORK_DIR=<repo> DATA_DIR=<shared data> \
#   OUT_ROOT=<results root> CKPT_DIR=<scratch weights dir> bash scripts/infra/olmo_ckpts.sh
set -uo pipefail
REPO="${REPO:-allenai/Olmo-3-1025-7B}"
CKPTS="${CKPTS:-stage1-step10000 stage1-step50000 stage1-step200000 stage1-step600000 stage1-step1413814 stage2-step47684 stage3-step11921}"
WORK="${WORK_DIR:?}"; DATA="${DATA_DIR:?}"; ROOT="${OUT_ROOT:?}"; CK="${CKPT_DIR:?}"
FAMS="${FAMS:-s0_zib,s0_quomp,s1_size,s1_loud,s1_heat}"
export PYTHONPATH="$WORK/src"
# N=12 core only (40/cell x 5 families + twins): the cell the readout ladder is reported on
[ -f "$DATA/core/stimuli.jsonl" ] || python3 "$WORK/scripts/generate_bcs.py" --out "$DATA/core" \
  --families "$FAMS" --n-grid 12 --per-cell 40 --difficulty hard --conditions shuffle
log(){ echo "[$(date '+%F %T')] $*"; }
for R in $CKPTS; do
  TAG="olmo3-7b@$R"; OUT="$ROOT/$TAG"
  if [ -f "$OUT/per_stim_rsa.done" ]; then log "skip $R (done)"; continue; fi
  log "CKPT START $R"
  python3 -c "from huggingface_hub import snapshot_download as s; s('$REPO', revision='$R', local_dir='$CK/$R', allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" \
    || { log "!! download failed $R"; continue; }
  OUT_ROOT="$ROOT" DATA_DIR="$DATA" MODEL_PATH="$CK/$R" ROLE=base EXTRACT_DS=core STAGES=gate,extract \
    bash "$WORK/run_scripts.sh" "$TAG"
  python3 "$WORK/scripts/per_stim_rsa.py" --acts "$OUT/acts" --model "$(echo "$TAG" | tr '/: ' '___')" --families "$FAMS" \
    --scheme readout --n-items 12 --pair-subsets onehop,multihop --stimuli "$DATA/core/stimuli.jsonl" --out "$OUT/dumps" \
    && touch "$OUT/per_stim_rsa.done"
  python3 "$WORK/scripts/summarize.py" "$OUT" "$DATA/core/questions.jsonl" | grep -E 'GATE|RSA' || true
  rm -rf "$CK/$R"
  log "CKPT END $R"
done
log "ALL CKPTS DONE"
