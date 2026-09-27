"""
Identify the species in ingested photographs — no labels required.

Runs MegaDetector first to find frames that contain an animal, then BioCLIP to
identify the species in those frames. Running the detector first matters:
BioCLIP always returns the nearest species on the candidate list, so without a
gate a photograph of waving grass comes back as a confident hyena.

Three ways to use it:

    --all                     identify among every configured species
    --species zebra           look for one species you already know
    --species zebra,wildebeest,buffalo
                              choose between a shortlist. Fewer candidates
                              means fewer things to confuse, so a shortlist
                              usually classifies better than the full list

Usage:
    python scripts/classify_images.py --all
    python scripts/classify_images.py --species zebra,wildebeest
    python scripts/classify_images.py --all --limit 200
    python scripts/classify_images.py --all --resume
"""

from __future__ import annotations

import argparse
import sys

from wildlife_monitor.config import TARGET_SPECIES
from wildlife_monitor.data.sufficiency import assess, guidance_for
from wildlife_monitor.db import (
    database_exists, load_detections, load_images, save_detections,
)
from wildlife_monitor.pipelines.classify import SpeciesClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                        help="identify among every configured species")
    group.add_argument("--species",
                        help="one species, or a comma-separated shortlist")
    parser.add_argument("--camera", help="restrict to one camera")
    parser.add_argument("--limit", type=int,
                        help="stop after this many images (for a trial run)")
    parser.add_argument("--resume", action="store_true",
                        help="skip images already classified")
    parser.add_argument("--no-gate", action="store_true",
                        help="skip the animal detector. Faster, but every "
                             "empty frame will be assigned a species")
    args = parser.parse_args()

    if not database_exists():
        print("[ERROR] No database yet. Run 'python scripts/init_db.py' and "
              "ingest some images first.")
        sys.exit(1)

    candidates = (list(TARGET_SPECIES) if args.all else
                  [name.strip().lower() for name in args.species.split(",")
                   if name.strip()])
    unknown = [name for name in candidates if name not in TARGET_SPECIES]
    if unknown:
        print(f"[WARN] Not in the configured species list: "
              f"{', '.join(unknown)}. They will still be used as prompts.")

    frame = load_images(args.camera,
                        unclassified_by="bioclip_classify" if args.resume else None)
    if frame.empty:
        print("[INFO] No images to classify."
              + (" All images already have results — drop --resume to redo them."
                 if args.resume else " Ingest some images first with "
                                     "scripts/ingest_images.py."))
        return
    if args.limit:
        frame = frame.head(args.limit)

    if args.no_gate:
        print("[WARN] Animal detector disabled. Empty frames will be given a "
              "species, and confidence will not reliably flag them.")

    classifier = SpeciesClassifier(candidates=candidates,
                                    use_gate=not args.no_gate)
    print(f"[INFO] {len(frame):,} images · {classifier.mode}")
    print("[INFO] Loading models ...")

    from wildlife_monitor.models import BioCLIPModel
    classifier.recogniser = BioCLIPModel()
    if not args.no_gate:
        from wildlife_monitor.models.megadetector import MegaDetector
        classifier.detector = MegaDetector()

    records, report = classifier.classify_frame(frame, path_column="file_path")

    print(f"\n{report.summary_line()}")
    if report.species_counts:
        print(f"\n{'species':<22}{'images':>8}")
        print("-" * 30)
        for species, count in sorted(report.species_counts.items(),
                                      key=lambda item: -item[1]):
            print(f"{species:<22}{count:>8,}")

    if report.empty_frames and report.empty_share > 0.5:
        print(f"\n[NOTE] {report.empty_share:.0%} of frames contained no "
              f"animal. That is normal for camera traps — most triggers are "
              f"wind or passing shadow — and those frames were dropped rather "
              f"than being assigned a species.")

    if not records:
        print("\n[DONE] Nothing to store.")
        return

    written = save_detections(records)
    print(f"\n[DONE] {written:,} detections stored.")

    coverage = assess(load_detections())
    print(f"\n{coverage.summary_line()}")
    for capability in coverage.capabilities:
        print(f"  {capability.name:<10}{capability.label}")
    guidance = guidance_for(coverage)
    if guidance:
        print(f"\n{guidance}")

    print("\nNext: train the behaviour models with")
    print("      python scripts/train_behaviour.py --all --model lstm")


if __name__ == "__main__":
    main()
