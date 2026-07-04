#!/usr/bin/env python
"""Step 4/4: pooled acts → curve fits → 2×2 quality readout → results/geometry.parquet.

CPU-only; runs on the DGX cores or any workstation.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
