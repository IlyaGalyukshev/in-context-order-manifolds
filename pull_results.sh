#!/usr/bin/env bash
# One command to pull results out of the persistent `manifolds` mlc_storage.
#
#   git pull && bash pull_results.sh              # results only (small, default)
#   git pull && bash pull_results.sh --with-acts  # + raw acts/ activations (~6+ GB)
#   bash pull_results.sh [--with-acts] <dest_dir> # custom dest (default ./results_dl)
#
# The mlc CLI has no storage-download command and results were written to the
# persistent storage (not job artifacts). This submits a tiny CPU job that mounts
# the storage read-only and re-publishes the files as job artifacts (see
# pull_preset*.yml), waits for it, then downloads them.
set -uo pipefail
cd "$(dirname "$0")"

PRESET="pull_preset.yml"
if [ "${1:-}" = "--with-acts" ]; then PRESET="pull_preset_full.yml"; shift; fi
DST="${1:-./results_dl}"
RE='manifolds-pull(-full)?-[A-Za-z0-9]+'

echo ">> submitting pull job ($PRESET)..."
SUB="$(mlc job submit --preset-file "$PRESET" --detach 2>&1)"
echo "$SUB"
NAME="$(printf '%s\n' "$SUB" | grep -oE "$RE" | head -1)"
if [ -z "${NAME:-}" ]; then
  NAME="$(mlc job ls -a 2>/dev/null | grep -oE "$RE" | head -1)"   # fallback: newest
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

# artifacts arrive as a .zip — unpack it in place so you get real files
if command -v unzip >/dev/null 2>&1; then
  find "$DST" -name '*.zip' -type f 2>/dev/null | while read -r z; do
    echo ">> unzip $(basename "$z")"; unzip -oq "$z" -d "$DST" && rm -f "$z"
  done
fi

N="$(find "$DST" -type f 2>/dev/null | wc -l | tr -d ' ')"
echo ">> $N files in $DST"
if [ "${N:-0}" = "0" ]; then
  echo "!! nothing came down — send me 'mlc job logs $NAME' and I'll fix the preset."
else
  find "$DST" -maxdepth 3 -type f 2>/dev/null | head -40
  echo "..."
  du -sh "$DST" 2>/dev/null
fi
