"""
Run a single detection pipeline.

Usage:
    python scripts/run_pipeline.py --pipeline bioclip_sam --species zebra
    python scripts/run_pipeline.py --pipeline sam3 --species elephant --top_n 20

Pipelines:
    bioclip_sam   BioCLIP retrieval + SAM 1 segmentation
    bioclip_yolo  BioCLIP retrieval + YOLOv11 bounding boxes
    sam3          SAM 3 text-prompted concept segmentation (SAM 1 fallback)
"""

import argparse

from wildlife_monitor.config import DEFAULT_TOP_N, TARGET_SPECIES
from wildlife_monitor.pipelines import PIPELINE_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one detection pipeline.")
    parser.add_argument("--pipeline", required=True,
                        choices=sorted(PIPELINE_REGISTRY.keys()),
                        help="Which pipeline to run.")
    parser.add_argument("--species", default="zebra",
                        help=f"Target species. One of: {', '.join(TARGET_SPECIES)}")
    parser.add_argument("--top_n", type=int, default=DEFAULT_TOP_N,
                        help="Number of images to process.")
    args = parser.parse_args()

    pipeline = PIPELINE_REGISTRY[args.pipeline]()
    pipeline.run(args.species, args.top_n)


if __name__ == "__main__":
    main()
