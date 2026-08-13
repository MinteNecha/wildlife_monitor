"""
Run all three pipelines and produce the comparison report.

Usage:
    python scripts/run_compare.py --species zebra --top_n 15
"""

import argparse

from wildlife_monitor.config import DEFAULT_TOP_N, TARGET_SPECIES
from wildlife_monitor.pipelines import PipelineComparator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and compare all three pipelines."
    )
    parser.add_argument("--species", default="zebra",
                        help=f"Target species. One of: {', '.join(TARGET_SPECIES)}")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N,
                        help="Number of images per pipeline.")
    args = parser.parse_args()

    PipelineComparator(args.species, args.top_n).run()


if __name__ == "__main__":
    main()
