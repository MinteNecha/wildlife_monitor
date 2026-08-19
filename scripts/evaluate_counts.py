"""
Pipeline 1 counting accuracy evaluation.

This script answers: which pipeline produces instance counts closest
to the citizen scientist ground truth?

It is a POST-PROCESSING evaluation only — it does not change how
pipelines work in production. In a real deployment, pipelines produce
instance_count from the detected animals and Pipeline 2 uses those
counts directly. Ground truth is only needed here to measure accuracy.

The script joins each pipeline's detections CSV with subset_metadata.csv
on image_id to retrieve the citizen scientist count, then computes:

  - Mean absolute error     (how far off each pipeline is on average)
  - Mean count              (is the pipeline over or undercounting?)
  - Exact match rate        (% of images where count exactly matches)
  - Count error by image    (so you can see which images are hardest)

Usage:
    python scripts/evaluate_counts.py
    python scripts/evaluate_counts.py --species buffalo zebra
    python scripts/evaluate_counts.py --species buffalo --top_n 15

Output:
    results/evaluation/count_accuracy_report.txt
    results/evaluation/count_accuracy_{species}.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from wildlife_monitor.config import RESULTS_DIR, SUBSET_CSV, TARGET_SPECIES

PIPELINES = ["bioclip_sam", "bioclip_yolo", "bioclip_megadetector"]
EVAL_DIR  = RESULTS_DIR / "evaluation"


def load_ground_truth() -> pd.DataFrame:
    """Load subset_metadata with ground truth count column."""
    csv_path = SUBSET_CSV
    if not csv_path.exists():
        alt = csv_path.parent.parent / "data" / "subset_metadata.csv"
        if alt.exists():
            csv_path = alt
    df = pd.read_csv(csv_path)
    if "ground_truth_count_num" not in df.columns:
        print("[ERROR] ground_truth_count_num not found in subset_metadata.csv.")
        print("        Run: python scripts/extract_ground_truth.py first.")
        return pd.DataFrame()
    return df[["image_id", "ground_truth_count_num",
               "ground_truth_count"]].rename(
        columns={"ground_truth_count_num": "gt_count"})


def evaluate_pipeline(pipeline: str, species: str,
                      gt: pd.DataFrame) -> pd.DataFrame | None:
    """Join one pipeline's detections against ground truth and compute error."""
    csv_path = RESULTS_DIR / pipeline / f"detections_{species}.csv"
    if not csv_path.exists():
        return None
    det = pd.read_csv(csv_path)
    if "instance_count" not in det.columns:
        return None
    merged = det.merge(gt, on="image_id", how="left")
    merged["count_error"] = abs(
        merged["instance_count"] - merged["gt_count"]
    )
    merged["exact_match"] = merged["instance_count"] == merged["gt_count"]
    merged["pipeline"] = pipeline
    return merged


def print_report(results: dict, species: str) -> str:
    lines = [
        "=" * 72,
        f"Pipeline 1 — Counting Accuracy Report  ({species})",
        "=" * 72,
        "",
        f"{'Pipeline':<28} {'Mean Count':>10} {'GT Count':>9}"
        f" {'MAE':>7} {'Exact %':>8}",
        "-" * 72,
    ]
    for pipeline, df in results.items():
        if df is None or df.empty:
            continue
        mean_pred = df["instance_count"].mean()
        mean_gt   = df["gt_count"].mean()
        mae       = df["count_error"].mean()
        exact     = df["exact_match"].mean() * 100
        lines.append(
            f"{pipeline:<28} {mean_pred:>10.1f} {mean_gt:>9.1f}"
            f" {mae:>7.1f} {exact:>7.1f}%"
        )
    lines += [
        "",
        "What these numbers mean:",
        "  Mean Count  = average number of animals detected by the pipeline",
        "  GT Count    = average citizen scientist ground truth count",
        "  MAE         = mean absolute error (lower is better)",
        "  Exact %     = % of images where pipeline count exactly matches GT",
        "",
        "Interpretation:",
        "  MAE < 2  = acceptable for social structure classification",
        "  MAE > 5  = pipeline is significantly over or undercounting",
        "=" * 72,
    ]
    return "\n".join(lines)


def run(species_list: list[str]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    if gt.empty:
        return

    for species in species_list:
        print(f"\n[INFO] Evaluating counting accuracy for '{species}' ...")
        results = {}
        for pipeline in PIPELINES:
            df = evaluate_pipeline(pipeline, species, gt)
            results[pipeline] = df
            if df is not None:
                mae   = df["count_error"].mean()
                exact = df["exact_match"].mean() * 100
                print(f"  {pipeline:<28} MAE={mae:.1f}  exact={exact:.1f}%")

        report = print_report(results, species)
        print("\n" + report)

        # Save report text
        report_path = EVAL_DIR / "count_accuracy_report.txt"
        report_path.write_text(report, encoding="utf-8")

        # Save per-image CSV
        all_rows = pd.concat(
            [d for d in results.values() if d is not None],
            ignore_index=True
        )
        if not all_rows.empty:
            out_csv = EVAL_DIR / f"count_accuracy_{species}.csv"
            all_rows[["image_id", "pipeline", "instance_count",
                       "gt_count", "ground_truth_count",
                       "count_error", "exact_match",
                       "confidence", "detection_quality"]].to_csv(
                out_csv, index=False)
            print(f"[INFO] Saved -> {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Pipeline 1 counting accuracy against ground truth"
    )
    parser.add_argument("--species", nargs="+", default=["buffalo"],
                        help="Species to evaluate (default: buffalo)")
    args = parser.parse_args()
    run(args.species)
