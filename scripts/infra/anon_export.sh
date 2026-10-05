#!/usr/bin/env bash
# Build an ANONYMISED copy of the repo for double-blind review (ARR / anonymous.4open.science).
#   bash scripts/infra/anon_export.sh [out_dir]      # default: ../anon_export
# Exports git HEAD (tracked files only), drops internal cluster presets, scrubs (python, portable) author names,
# e-mail, private paths, internal registries and personal dataset ids, then FAILS if any
# identifying string survives. Also writes <out_dir>.zip for supplementary material.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-../anon_export}"
rm -rf "$OUT" && mkdir -p "$OUT"
git archive HEAD | tar -x -C "$OUT"

# internal cluster tooling: not needed to reproduce, and identifies infrastructure
rm -f "$OUT"/.ml-job-preset.yml "$OUT"/steer_preset.yml "$OUT"/pull_preset.yml "$OUT"/pull_preset_full.yml "$OUT"/pull_results.sh

python3 - "$OUT" <<'PYSCRUB'
import os, re, sys
root = sys.argv[1]
R = [  # (pattern, replacement) — order matters
    (r"Ilya Galyukshev", "Anonymous Authors"),
    (r"galyukshev\.ilya@gmail\.com", "anonymous@example.org"),
    (r"github\.com/IlyaGalyukshev/", "anonymous.4open.science/r/"),
    (r"IlyaGalyukshev", "anonymous"),
    (r"galyukshev/in-context-order-manifolds", "ANON/in-context-order-manifolds"),
    (r"/Users/[\w.-]+/[^\"' ]*?/(data/)", r"\1"),
    (r"/Users/[\w.-]+/", "~/"),
    (r"[a-z.-]*artifactory\.tcsbank\.ru/[^\s\"']*", "<registry>/<image>"),
    (r"ilya_mfld", "mfld"), (r"ilya_workspace", "workspace"), (r"ilya_", "user_"),
    (r"Ilya's", "the authors'"), (r"\bIlya\b", "the authors"),
]
for dp, _, fs in os.walk(root):
    for f in fs:
        fp = os.path.join(dp, f)
        try:
            t = open(fp, encoding="utf-8").read()
        except Exception:
            continue                                   # binary file
        u = t
        for a, b in R:
            u = re.sub(a, b, u)
        if u != t:
            open(fp, "w", encoding="utf-8").write(u)
PYSCRUB

LEFT=$(grep -rnIiE 'galyukshev|ilya|tcsbank|artifactory|/Users/' "$OUT" || true)
if [ -n "$LEFT" ]; then
  echo "!! identifying strings remain:"; echo "$LEFT" | head -40; exit 1
fi
( cd "$(dirname "$OUT")" && rm -f "$(basename "$OUT").zip" && zip -qr "$(basename "$OUT").zip" "$(basename "$OUT")" )
echo "anonymised export OK -> $OUT  (+ $OUT.zip, $(find "$OUT" -type f | wc -l | tr -d ' ') files)"
