#!/usr/bin/env python
"""Step 3/4: one HF forward pass per stimulus → pooled acts/*.npz (never full tensors)."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
