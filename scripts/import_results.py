"""
Import existing CSV results into the database.

The pipelines now write to SQLite, but detection runs completed before the
migration live in results/<pipeline>/detections_<species>.csv and represent
hours of compute. This script loads them into the normalised schema so nothing
already produced has to be recomputed.

Importing is idempotent. Detection carries UNIQUE (image_id, pipeline_id), so
re-importing the same CSV updates those rows rather than duplicating them.

Usage:
    python scripts/import_results.py                    # every CSV found
    python scripts/import_results.py --species zebra
    python scripts/import_results.py --pipeline bioclip_megadetector
    python scripts/import_results.py --subset            # also camera metadata
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from wildlife_monitor.config import RESULTS_DIR, SUBSET_CSV
from wildlife_monitor.db import Database, init_db, table_counts

REQUIRED_COLUMNS = {"image_id", "species", "camera_id"}


def find_csvs(pipeline: str | None, species: str | None) -> list[Path]:
    """Every detections CSV matching the filters, in a stable order."""
    directories = ([RESULTS_DIR / pipeline] if pipeline
                   else [path for path in sorted(RESULTS_DIR.glob("*"))
                         if path.is_dir()])
    pattern = f"detections_{species}.csv" if species else "detections_*.csv"
    return [csv for directory in directories
            if directory.exists()
            for csv in sorted(directory.glob(pattern))]


def import_csv(database: Database, csv_path: Path) -> tuple[int, int]:
    """Import one CSV, returning ``(imported, skipped)``."""
    frame = pd.read_csv(csv_path)
    if frame.empty:
        return 0, 0

    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        print(f"  [SKIP] {csv_path.name}: missing columns {sorted(missing)}")
        return 0, len(frame)

    # The pipeline column may be absent in older files; fall back to the
    # directory name, which is where the pipeline wrote its output.
    if "pipeline" not in frame.columns:
        frame["pipeline"] = csv_path.parent.name

    imported = skipped = 0
    for record in frame.to_dict(orient="records"):
        if not str(record.get("image_id", "")).strip():
            skipped += 1
            continue
        try:
            database.detections.save_record(record)
            imported += 1
        except Exception as error:
            skipped += 1
            if skipped <= 3:
                print(f"  [WARN] row skipped: {error}")
    return imported, skipped


def import_subset(database: Database, path: Path) -> int:
    """Enrich Camera rows from the subset metadata (GPS, habitat, season)."""
    if not path.exists():
        print(f"[INFO] No subset metadata at {path}; skipping camera enrichment.")
        return 0
    frame = pd.read_csv(path)
    if "site_id" not in frame.columns:
        print("[INFO] Subset metadata has no site_id column; skipping.")
        return 0

    sites = frame.drop_duplicates(subset=["site_id"])
    for _, row in sites.iterrows():
        database.cameras.ensure(
            row["site_id"], row.get("latitude", 0.0), row.get("longitude", 0.0),
            row.get("habitat_type", "unknown"))
    return len(sites)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", help="only this pipeline's results")
    parser.add_argument("--species", help="only this species")
    parser.add_argument("--subset", action="store_true",
                        help="also import camera metadata from the subset CSV")
    args = parser.parse_args()

    init_db()
    csvs = find_csvs(args.pipeline, args.species)
    if not csvs:
        print(f"[ERROR] No detections CSVs found under {RESULTS_DIR}")
        return

    print(f"Found {len(csvs)} CSV file(s) to import.\n")
    total_imported = total_skipped = 0
    with Database() as database:
        if args.subset:
            sites = import_subset(database, SUBSET_CSV)
            print(f"[OK] Camera metadata for {sites} sites\n")

        for csv_path in csvs:
            imported, skipped = import_csv(database, csv_path)
            total_imported += imported
            total_skipped += skipped
            relative = csv_path.relative_to(RESULTS_DIR)
            print(f"  {str(relative):<52}{imported:>7,} imported"
                  + (f"  ({skipped} skipped)" if skipped else ""))

    print(f"\n[DONE] {total_imported:,} detections imported"
          + (f", {total_skipped:,} skipped" if total_skipped else ""))

    print(f"\n{'table':<22}{'rows':>10}")
    print("-" * 32)
    for table, count in table_counts().items():
        if count:
            print(f"{table:<22}{count:>10,}")


if __name__ == "__main__":
    main()
