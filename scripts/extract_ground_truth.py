"""
Extract ground truth count and behavioural annotations from the
Snapshot Serengeti Season 1 annotation JSON and add them to
subset_metadata.csv.

Fields added to subset_metadata.csv:
  ground_truth_count      — citizen scientist count string e.g. "1","11-50"
  ground_truth_count_num  — numeric midpoint (1-10 exact, 11-50->15, 51+->51)
  gt_standing             — proportion of annotators who said standing
  gt_resting              — proportion who said resting
  gt_moving               — proportion who said moving
  gt_interacting          — proportion who said interacting
  gt_young_present        — proportion who said young present

These fields serve two purposes:
  1. Pipeline 1 evaluation — compare instance_count against ground_truth_count_num
  2. Pipeline 2 training labels — gt_standing/resting/moving/interacting encode
     the citizen scientist behavioural observations that the temporal model
     is trained to predict from detection sequences.

Usage:
    python scripts/extract_ground_truth.py
    python scripts/extract_ground_truth.py --json path/to/annotations.json
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import pandas as pd

from wildlife_monitor.config import PROJECT_ROOT, SUBSET_CSV

DEFAULT_ANN_PATH = (
    PROJECT_ROOT / "data" / "SnapshotSerengetiS01.json"
)

COUNT_MAP = {
    "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    "6": 6, "7": 7, "8": 8, "9": 9, "10": 10,
    "11-50": 15, "51+": 51,
}


def run(ann_path: Path) -> None:
    if not ann_path.exists():
        print(f"[ERROR] Annotation file not found: {ann_path}")
        print("Set --json to the full path of SnapshotSerengetiS01.json")
        return

    csv_path = SUBSET_CSV
    if not csv_path.exists():
        alt = PROJECT_ROOT.parent / "data" / "subset_metadata.csv"
        if alt.exists():
            csv_path = alt
        else:
            print(f"[ERROR] subset_metadata.csv not found.")
            return

    print(f"Loading annotation JSON ({ann_path.stat().st_size // 1_000_000} MB) ...")
    with open(ann_path, encoding="utf-8") as f:
        data = json.load(f)

    anns = data.get("annotations", [])
    print(f"Total annotations: {len(anns):,}")

    # Build image_id -> annotation lookup (one annotation per image)
    print("Building lookup ...")
    lookup: dict[str, dict] = {}
    for ann in anns:
        img_id = ann.get("image_id")
        if img_id:
            lookup[img_id] = ann
    print(f"Unique image IDs: {len(lookup):,}")

    df = pd.read_csv(csv_path)
    print(f"\nSubset rows: {len(df)}")

    gt_count, gt_count_num = [], []
    gt_stand, gt_rest, gt_move, gt_inter, gt_young = [], [], [], [], []

    matched = 0
    for _, row in df.iterrows():
        ann = lookup.get(str(row["image_id"]))
        if ann:
            matched += 1
            c = str(ann.get("count", "1"))
            gt_count.append(c)
            gt_count_num.append(COUNT_MAP.get(c, 1))
            gt_stand.append(ann.get("standing", 0.0))
            gt_rest.append(ann.get("resting", 0.0))
            gt_move.append(ann.get("moving", 0.0))
            gt_inter.append(ann.get("interacting", 0.0))
            gt_young.append(ann.get("young_present", 0.0))
        else:
            gt_count.append(None); gt_count_num.append(None)
            gt_stand.append(None); gt_rest.append(None)
            gt_move.append(None); gt_inter.append(None)
            gt_young.append(None)

    df["ground_truth_count"]     = gt_count
    df["ground_truth_count_num"] = gt_count_num
    df["gt_standing"]            = gt_stand
    df["gt_resting"]             = gt_rest
    df["gt_moving"]              = gt_move
    df["gt_interacting"]         = gt_inter
    df["gt_young_present"]       = gt_young

    print(f"Matched {matched}/{len(df)} rows ({matched/len(df)*100:.1f}%)")

    # Sample output
    sample = df[df["species_label"] == "buffalo"][
        ["image_id", "species_label",
         "ground_truth_count", "ground_truth_count_num",
         "gt_standing", "gt_moving"]
    ].head(5)
    print(f"\nSample buffalo rows:\n{sample.to_string()}")

    print("\nGround truth count distribution (all species):")
    print(df["ground_truth_count"].value_counts().head(15))

    df.to_csv(csv_path, index=False)
    print(f"\n[INFO] Saved updated subset_metadata.csv -> {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=DEFAULT_ANN_PATH,
                        help="Path to SnapshotSerengetiS01.json")
    args = parser.parse_args()
    run(args.json)
