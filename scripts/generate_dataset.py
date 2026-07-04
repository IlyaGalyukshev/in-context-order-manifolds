#!/usr/bin/env python
"""Step 1/4: generate stimuli + question batteries from configs/generation.yaml."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
