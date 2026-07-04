#!/usr/bin/env bash
# Start a GPU worker container on a remote docker context.
# Usage: CONTEXT=<ctx> GPUS=0,1 WORKSPACE=/path/on/host ./start_worker.sh
# Notes for our fleet: some hosts need DOCKER_API_VERSION=1.43; --gpus does not
# work — use the nvidia runtime + NVIDIA_VISIBLE_DEVICES. Always check
# nvidia-smi for free GPUs before claiming any.
set -euo pipefail

: "${CONTEXT:?set CONTEXT (docker context name)}"
: "${GPUS:?set GPUS (e.g. 0,1,2)}"
: "${WORKSPACE:?set WORKSPACE (host path to the shared workspace)}"
NAME="${NAME:-ilya_mfld_worker}"
IMAGE="${IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"

docker --context "$CONTEXT" run -d --name "$NAME" \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  --shm-size=32g \
  -v "$WORKSPACE":/workspace \
  -e HF_HOME=/workspace/manifolds/hf_cache \
  "$IMAGE" sleep infinity

echo "Worker $NAME up on $CONTEXT (GPUs $GPUS)."
echo "Next: docker --context $CONTEXT exec -it $NAME bash -c 'pip install -e /workspace/manifolds/repo[gpu,battery,dev]'"
