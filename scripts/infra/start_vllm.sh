#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for the behavioral battery.
# v0.6.6 is pinned: the last line with solid V100 (SM 7.0) support.
# Usage: CONTEXT=<ctx> GPUS=4,5 WORKSPACE=/path MODEL=meta-llama/Llama-3.1-8B-Instruct ./start_vllm.sh
set -euo pipefail

: "${CONTEXT:?set CONTEXT}"
: "${GPUS:?set GPUS}"
: "${WORKSPACE:?set WORKSPACE}"
: "${MODEL:?set MODEL (hf id)}"
PORT="${PORT:-8901}"
TP="${TP:-$(awk -F, '{print NF}' <<< "$GPUS")}"
NAME="${NAME:-ilya_mfld_vllm}"

docker --context "$CONTEXT" run -d --name "$NAME" \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  --shm-size=32g -p "$PORT":8000 \
  -v "$WORKSPACE"/manifolds/hf_cache:/root/.cache/huggingface \
  vllm/vllm-openai:v0.6.6.post1 \
  --model "$MODEL" --dtype half --max-model-len 8192 \
  --enable-prefix-caching --tensor-parallel-size "$TP"

echo "vLLM $MODEL on $CONTEXT:$PORT (TP=$TP). Stop with: docker --context $CONTEXT rm -f $NAME"
