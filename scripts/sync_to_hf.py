#!/usr/bin/env python
"""Sync a results directory to the canonical HuggingFace dataset.

    python scripts/sync_to_hf.py <results_dir> [--name MODEL] [--acts] \
        [--repo galyukshev/in-context-order-manifolds]

`<results_dir>` is a per-model results dir — the probe/RSA JSONs, `battery/`,
`acts_mean/`, figures — as written by run_scripts.sh under OUT_ROOT/<TAG>. It is
uploaded to `results/<MODEL>/` in the dataset (MODEL defaults to the dir
basename). The multi-GB raw `acts/` (per-stimulus activation reads) are skipped
unless `--acts`; `acts_mean/` (the small k-mean arrays needed for cPCA/decode)
always goes.

HF Hub is reachable from the DGX and locally, so run this wherever results land —
one command, no cross-machine file shuffling. This is the canonical store we
cite in the paper (Data Availability). Needs a write token (`hf auth login`).
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="per-model results dir (contains the JSONs, battery/, acts_mean/)")
    ap.add_argument("--name", default=None, help="model name in the dataset (default: dir basename)")
    ap.add_argument("--repo", default="galyukshev/in-context-order-manifolds")
    ap.add_argument("--acts", action="store_true", help="also upload the multi-GB raw acts/ activations")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo

    d = Path(args.results_dir).resolve()
    if not d.is_dir():
        raise SystemExit(f"not a directory: {d}")
    name = args.name or d.name

    create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    ignore = None if args.acts else ["acts/**", "**/acts/**", "*.zip", "*.zip.tmp"]

    print(f">> uploading {d} -> {args.repo}:results/{name}  (raw acts: {'yes' if args.acts else 'no'})")
    HfApi().upload_folder(
        folder_path=str(d), repo_id=args.repo, repo_type="dataset",
        path_in_repo=f"results/{name}", ignore_patterns=ignore,
        commit_message=f"sync results/{name}")
    print(f">> done: https://huggingface.co/datasets/{args.repo}/tree/main/results/{name}")


if __name__ == "__main__":
    main()
