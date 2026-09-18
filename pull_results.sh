#!/usr/bin/env bash
# One command to pull ALL results out of the persistent `manifolds` mlc_storage.
#
#   git pull && bash pull_results.sh
#
# The mlc CLI has no storage-download command, and results were written to the
# persistent storage (not job artifacts). So this submits a tiny CPU job that
# mounts the storage read-only and re-publishes its result files (except the big
# acts/ npz) as job artifacts (see pull_preset.yml), waits for it, then downloads
# everything into ./results_dl. Pass a different dest dir as the first arg.
set -uo pipefail
cd "$(dirname "$0")"

DST="${1:-./results_dl}"
PRESET="pull_preset.yml"

echo ">> submitting pull job (mounts manifolds, publishes results as artifacts)..."
SUB="$(mlc job submit --preset-file "$PRESET" --detach 2>&1)"
echo "$SUB"
NAME="$(printf '%s\n' "$SUB" | grep -oE 'manifolds-pull-[A-Za-z0-9]+' | head -1)"
if [ -z "${NAME:-}" ]; then
  NAME="$(mlc job ls -a 2>/dev/null | grep -oE 'manifolds-pull-[A-Za-z0-9]+' | head -1)"   # fallback: newest
fi
if [ -z "${NAME:-}" ]; then
  echo "!! could not determine job name — check 'mlc job ls -a'"; exit 1
fi
echo ">> job: $NAME"

echo ">> waiting for it to finish (polls every 15s)..."
while true; do
  ST="$(mlc job get "$NAME" 2>/dev/null | grep -ioE 'SUCCEEDED|FAILED|CANCELLED|TIMEOUT|RUNNING|PENDING' | head -1)"
  echo "   state: ${ST:-?}"
  case "${ST:-}" in
    SUCCEEDED) break ;;
    FAILED|CANCELLED|TIMEOUT) echo "!! job ended: $ST — see 'mlc job logs $NAME'"; exit 1 ;;
  esac
  sleep 15
done

echo ">> downloading artifacts -> $DST"
mlc job download artifacts "$NAME" --dst-folder "$DST"

N="$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo ">> downloaded $N files into $DST"
if [ "${N:-0}" = "0" ]; then
  echo "!! no files came down — the output/artifacts step didn't capture the storage."
  echo "   ping me (Claude) with 'mlc job logs $NAME' output and I'll fix the preset."
else
  find "$DST" -maxdepth 3 -type f 2>/dev/null | head -40
  du -sh "$DST" 2>/dev/null
fi
