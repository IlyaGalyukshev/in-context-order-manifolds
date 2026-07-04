#!/usr/bin/env python
"""Step 2/4: behavioral battery against a running vLLM server → results/*.parquet."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
