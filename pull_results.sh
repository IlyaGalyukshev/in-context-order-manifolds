#!/usr/bin/env bash
# Pull results out of the persistent `manifolds` mlc_storage (no storage-download
# command in mlc; results live in the storage, not job artifacts).
#
#   git pull && bash pull_results.sh                        # results only (small)
#   bash pull_results.sh --with-acts                        # + raw acts/ (~6+ GB)
#   bash pull_results.sh --download-only <job-name>         # re-download a finished job's
#                                                           #   artifacts (retries, no re-run)
#   bash pull_results.sh [flags] <dest_dir>                 # dest (default ./results_dl)
#
# Downloads land in ./results_dl (a subdir of the repo), as a .zip — NOT unzipped.
# The download retries on connection resets (big single-zip transfers over a
# flaky proxy drop mid-way; mlc download has no resume, so we restart the attempt).
set -uo pipefail
cd "$(dirname "$0")"

PRESET="pull_preset.yml"
JOB=""
DST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --with-acts)     PRESET="pull_preset_full.yml"; shift ;;
    --download-only) JOB="${2:-}"; shift 2 ;;
    *)               DST="$1"; shift ;;
  esac
done
DST="${DST:-./results_dl}"
RE='manifolds-pull(-full)?-[A-Za-z0-9]+'

if [ -z "$JOB" ]; then
  echo ">> submitting pull job ($PRESET)..."
  SUB="$(mlc job submit --preset-file "$PRESET" --detach 2>&1)"; echo "$SUB"
  JOB="$(printf '%s\n' "$SUB" | grep -oE "$RE" | head -1)"
  [ -z "$JOB" ] && JOB="$(mlc job ls -a 2>/dev/null | grep -oE "$RE" | head -1)"
  [ -z "$JOB" ] && { echo "!! could not determine job name — check 'mlc job ls -a'"; exit 1; }
  echo ">> job: $JOB"
  echo ">> waiting for it to finish (polls every 15s)..."
  while true; do
    ST="$(mlc job get "$JOB" 2>/dev/null | grep -ioE 'SUCCEEDED|FAILED|CANCELLED|TIMEOUT|RUNNING|PENDING' | head -1)"
    echo "   state: ${ST:-?}"
    case "${ST:-}" in
      SUCCEEDED) break ;;
      FAILED|CANCELLED|TIMEOUT) echo "!! job ended: $ST — see 'mlc job logs $JOB'"; exit 1 ;;
    esac
    sleep 15
  done
fi

echo ">> downloading artifacts of $JOB -> $DST (retries on reset)"
mkdir -p "$DST"
ok=0
for attempt in 1 2 3 4 5 6 7 8; do
  echo "   attempt $attempt ..."
  rm -f "$DST"/*.zip.tmp 2>/dev/null
  if mlc job download artifacts "$JOB" --dst-folder "$DST"; then ok=1; break; fi
  echo "   ...reset; retrying in 10s"; sleep 10
done

echo
if [ "$ok" != 1 ]; then
  echo "!! download kept failing — the single 6 GB zip won't survive this link."
  echo "   ping me (Claude); I'll switch the full pull to ~1 GB chunks you can grab one by one."
  exit 1
fi
echo ">> done. in $DST :"
find "$DST" -maxdepth 2 -type f 2>/dev/null | tail -30
du -sh "$DST" 2>/dev/null
find "$DST" -name '*.zip' -type f 2>/dev/null | while read -r z; do
  echo ">> archive: $z  — unpack with:  unzip -d \"$DST\" \"$z\""
done
