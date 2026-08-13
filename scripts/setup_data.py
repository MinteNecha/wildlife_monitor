"""
One-time environment and data setup.

Verifies the project directories exist, checks that the subset metadata is
present, downloads the SAM 1 checkpoint, and reports whether the optional
SAM 3 weights are available. Run this once before the pipelines.

Usage:
    python scripts/setup_data.py
"""

from wildlife_monitor.config import (
    ensure_directories, SUBSET_CSV, SAM3_CHECKPOINT, TARGET_SPECIES,
)
from wildlife_monitor.data import DatasetLoader
from wildlife_monitor.models import ensure_sam1_checkpoint, sam3_weights_available


def main() -> None:
    print("=" * 60)
    print("  Wildlife Monitor — environment setup")
    print("=" * 60)

    ensure_directories()
    print("[OK] Project directories ready.")

    # SAM 1 checkpoint (used by BioCLIP+SAM and the SAM 3 fallback).
    ensure_sam1_checkpoint()
    print("[OK] SAM 1 checkpoint ready.")

    # Subset metadata.
    if SUBSET_CSV.exists():
        DatasetLoader().prepare()
        print("[OK] Subset metadata found.")
    else:
        print(f"[ACTION NEEDED] Place your prepared subset_metadata.csv at:")
        print(f"                {SUBSET_CSV}")
        print(f"                and the images under data/images/.")

    # Optional SAM 3 weights.
    if sam3_weights_available():
        print("[OK] SAM 3 weights found — text-prompted mode available.")
    else:
        print("[INFO] SAM 3 weights not found (optional).")
        print("       The SAM 3 pipeline will fall back to SAM 1.")
        print("       To enable SAM 3:")
        print("         1. pip install -U ultralytics   (>= 8.3.237)")
        print("         2. Request access + download sam3.pt from")
        print("            https://huggingface.co/facebook/sam3")
        print(f"         3. Place sam3.pt at {SAM3_CHECKPOINT}")

    print("\nSetup complete. Target species:")
    print("  " + ", ".join(TARGET_SPECIES))


if __name__ == "__main__":
    main()
