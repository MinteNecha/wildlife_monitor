"""
Ingest a folder of camera trap photographs — no species labels required.

Expects one folder per camera:

    photos/
      SiteA/  IMG_0001.JPG  ...
      SiteB/  ...

The folder name becomes the camera. Capture times are read from EXIF.
Location and habitat come from a camera file you supply:

    camera_id,latitude,longitude,habitat_type
    SiteA,-2.15,34.80,woodland
    SiteB,-2.21,34.85,open_grassland

Run with --make-cameras first and the script writes that file for you,
pre-filled with the camera names it found, so the format cannot be wrong.

Usage:
    python scripts/ingest_images.py --images photos/ --make-cameras
    python scripts/ingest_images.py --images photos/ --cameras cameras.csv
    python scripts/ingest_images.py --images photos/ --cameras cameras.csv --prepare
    python scripts/ingest_images.py --images photos/ --cameras cameras.csv --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from wildlife_monitor.config import DATA_DIR
from wildlife_monitor.data.cameras import (
    CameraRegistry, blocking_issues, warnings,
)
from wildlife_monitor.data.ingestion import ImageIngestor, discover_cameras
from wildlife_monitor.data.preparation import ImagePreparer
from wildlife_monitor.data.sufficiency import assess, guidance_for
from wildlife_monitor.data.validator import ImageValidator
from wildlife_monitor.db import init_db, load_detections

DEFAULT_CAMERAS = DATA_DIR / "cameras.csv"


def write_camera_template(root: Path, path: Path) -> int:
    """Write a camera file pre-filled with the folders that were found."""
    cameras = discover_cameras(root)
    if not cameras:
        print(f"[ERROR] No images found under {root}")
        return 0
    registry = CameraRegistry(CameraRegistry.template(sorted(cameras)))
    registry.write(path)
    print(f"[OK] Wrote {path} with {len(cameras)} camera(s):")
    for camera_id, images in sorted(cameras.items()):
        print(f"       {camera_id:<20} {len(images):>6,} images")
    print("\nFill in latitude, longitude and habitat_type, then run again "
          "with --cameras.")
    print("Habitats: open_grassland, woodland, riverine, kopje, unknown")
    return len(cameras)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True,
                        help="folder containing one sub-folder per camera")
    parser.add_argument("--cameras", default=str(DEFAULT_CAMERAS),
                        help=f"camera metadata CSV (default {DEFAULT_CAMERAS})")
    parser.add_argument("--make-cameras", action="store_true",
                        help="write a pre-filled camera file and exit")
    parser.add_argument("--prepare", action="store_true",
                        help="convert formats, fix rotation, shrink large images")
    parser.add_argument("--allow-upscale", action="store_true",
                        help="also enlarge undersized images. This adds no "
                             "detail; such images are flagged in the database")
    parser.add_argument("--min-width", type=int, default=640)
    parser.add_argument("--min-height", type=int, default=480)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would happen without writing")
    args = parser.parse_args()

    root = Path(args.images)
    if not root.exists():
        print(f"[ERROR] No such folder: {root}")
        sys.exit(1)

    cameras_path = Path(args.cameras)
    if args.make_cameras:
        write_camera_template(root, cameras_path)
        return

    if not cameras_path.exists():
        print(f"[ERROR] No camera file at {cameras_path}.")
        print(f"        Create one with:  python scripts/ingest_images.py "
              f"--images {root} --make-cameras")
        sys.exit(1)

    registry = CameraRegistry.load(cameras_path)
    found = sorted(discover_cameras(root))
    issues = registry.validate(known_cameras=found)

    for issue in warnings(issues):
        print(f"[WARN]  {issue.message}")
    blocking = blocking_issues(issues)
    if blocking:
        for issue in blocking:
            print(f"[ERROR] {issue.message}")
        print("\nFix the camera file and run again.")
        sys.exit(1)

    validator = ImageValidator(min_width=args.min_width,
                                min_height=args.min_height)
    preparer = None
    if args.prepare or args.allow_upscale:
        preparer = ImagePreparer(min_width=args.min_width,
                                  min_height=args.min_height,
                                  allow_upscale=args.allow_upscale)
        print(f"[INFO] Preparation enabled: {preparer.description}")

    init_db()
    report = ImageIngestor(validator, preparer).ingest(
        root, registry, dry_run=args.dry_run)

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}{report.summary_line()}")
    if report.first_capture:
        print(f"          Covering {report.first_capture[:10]} to "
              f"{report.last_capture[:10]}")

    if report.rejected:
        print(f"\nRejected {report.rejected_count} image(s):")
        for name, reason in report.rejected[:10]:
            print(f"    {name}: {reason}")
        if report.rejected_count > 10:
            print(f"    ... and {report.rejected_count - 10} more")
        if preparer is None:
            print("\n  Some of these may be fixable. Re-run with --prepare to "
                  "convert formats and fix rotation, or lower the threshold "
                  "with --min-width / --min-height.")

    if report.upscaled:
        print(f"\n[NOTE] {report.upscaled} image(s) were enlarged to pass the "
              f"resolution check. Enlarging adds no detail, so detection on "
              f"these will be no better than at their original size. They are "
              f"flagged in the database so their results stay identifiable.")

    if report.undated:
        print(f"\n[NOTE] {report.undated} image(s) had no EXIF capture time. "
              f"Behavioural analysis needs timestamps, so those images can be "
              f"classified but not used for behaviour.")

    if not args.dry_run and report.ingested:
        print("\nNext: identify the species in these images with")
        print("      python scripts/classify_images.py --all")


if __name__ == "__main__":
    main()
