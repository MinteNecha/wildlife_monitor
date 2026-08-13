"""
BioCLIP vs SAM 3 — direct model comparison.

This script answers Research Question 1 directly:
    Which pipeline best combines accuracy, localisation quality, and speed?

It evaluates three things side by side:

    BioCLIP Identification Accuracy
        For each image, BioCLIP is scored against ALL species prompts
        simultaneously. The species with the highest cosine similarity wins.
        Correct = the winning species matches the ground-truth label.
        This is exactly how the 43.7% top-1 accuracy figure was produced.

    SAM 3 Localisation Rate
        SAM 3 is given the correct species name as a text prompt.
        Its job is to FIND and SEGMENT the animal.
        Correct = SAM 3 produced a valid mask (not "no_mask") for that image.
        Since it already knows the species name, its accuracy question is:
        "Can it locate the animal from a text description?"

    BioCLIP + SAM 3 Combined
        Uses BioCLIP's top-1 prediction AND SAM 3 localisation together.
        An image is fully correct only if BioCLIP identified it right AND
        SAM 3 successfully segmented it.

The output shows per-species accuracy for all three, plus a clear
comparison table showing which pipeline performed best on each species.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --species zebra buffalo
    python scripts/compare_models.py --top_n 15

Output:
    results/evaluation/bioclip_vs_sam3_results.csv
    results/evaluation/bioclip_vs_sam3_summary.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from wildlife_monitor.config import (
    SUBSET_CSV, RESULTS_DIR, PROMPT_TEMPLATE, TARGET_SPECIES, SystemConfig
)
from wildlife_monitor.data.loader import _resolve_image_path
from wildlife_monitor.models import BioCLIPModel, SAM3Segmenter

OUTPUT_DIR = RESULTS_DIR / "evaluation"


# ── Step 1: BioCLIP multi-species identification ───────────────────────────────

def run_bioclip_evaluation(
    eval_frame: pd.DataFrame,
    target_species: list[str],
    bioclip: BioCLIPModel,
) -> tuple[dict[str, dict], float]:
    """
    Score every image against all species prompts simultaneously.
    Returns {image_id: {predicted_species, correct, score, true_species}}
    and elapsed seconds.
    """
    print("\n[BioCLIP] Computing text embeddings for all species prompts...")
    text_feats = {}
    for sp in target_species:
        prompt = PROMPT_TEMPLATE.format(species=sp)
        tokens = bioclip.tokenizer([prompt]).to(bioclip.device)
        with torch.no_grad():
            feat = bioclip.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        text_feats[sp] = feat
    print(f"[BioCLIP] {len(text_feats)} prompts encoded. Scoring images...")

    results = {}
    start = time.time()

    for _, row in tqdm(eval_frame.iterrows(), total=len(eval_frame),
                       desc="BioCLIP scoring"):
        img_path = Path(row["local_image_path"])
        true_species = str(row["species_label"])

        try:
            img_tensor = bioclip.preprocess(
                Image.open(img_path).convert("RGB")
            ).unsqueeze(0).to(bioclip.device)
            with torch.no_grad():
                img_feat = bioclip.model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        except Exception as e:
            print(f"\n  [WARN] Could not read {img_path.name}: {e}")
            continue

        scores = {sp: float((img_feat @ text_feats[sp].T).squeeze())
                  for sp in target_species}
        predicted = max(scores, key=scores.get)
        correct = predicted == true_species

        results[str(row["image_id"])] = {
            "true_species":      true_species,
            "bioclip_predicted": predicted,
            "bioclip_correct":   correct,
            "bioclip_score":     round(scores[predicted], 4),
            "bioclip_true_score": round(scores.get(true_species, 0.0), 4),
            "image_path":        str(img_path),
        }

    elapsed = time.time() - start
    return results, elapsed


# ── Step 2: SAM 3 localisation ────────────────────────────────────────────────

def run_sam3_evaluation(
    eval_frame: pd.DataFrame,
    bioclip_results: dict[str, dict],
) -> tuple[dict[str, dict], float]:
    """
    For each image, run SAM 3 with the correct species text prompt.
    Records whether SAM 3 found a valid mask and its quality score.
    """
    # Group images by species so we only load each SAM3Segmenter once per species
    species_groups = eval_frame.groupby("species_label")
    sam3_results = {}
    total_elapsed = 0.0

    for species, group in species_groups:
        print(f"\n[SAM 3] Processing {len(group)} images of '{species}'...")
        segmenter = SAM3Segmenter(species)
        print(f"        Backend: {segmenter.backend}")

        start = time.time()
        for _, row in tqdm(group.iterrows(), total=len(group),
                           desc=f"SAM 3 · {species}"):
            img_path = Path(row["local_image_path"])
            image_id = str(row["image_id"])

            try:
                import cv2
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    raise ValueError("imread returned None")
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"\n  [WARN] Could not read {img_path.name}: {e}")
                sam3_results[image_id] = {
                    "sam3_localised": False,
                    "sam3_quality":   0.0,
                    "sam3_backend":   getattr(segmenter, "backend", "unknown"),
                }
                continue

            mask, quality = segmenter.segment(image_rgb)
            localised = mask is not None and quality > 0.0

            sam3_results[image_id] = {
                "sam3_localised": localised,
                "sam3_quality":   round(quality, 4),
                "sam3_backend":   segmenter.backend,
            }

        total_elapsed += time.time() - start
        del segmenter

    return sam3_results, total_elapsed


# ── Step 3: Merge and report ──────────────────────────────────────────────────

def build_combined_results(
    bioclip_results: dict[str, dict],
    sam3_results: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for image_id, bc in bioclip_results.items():
        s3 = sam3_results.get(image_id, {
            "sam3_localised": False, "sam3_quality": 0.0, "sam3_backend": "unknown"
        })
        combined_correct = bc["bioclip_correct"] and s3["sam3_localised"]
        rows.append({
            "image_id":           image_id,
            "image_path":         bc["image_path"],
            "true_species":       bc["true_species"],
            "bioclip_predicted":  bc["bioclip_predicted"],
            "bioclip_correct":    bc["bioclip_correct"],
            "bioclip_score":      bc["bioclip_score"],
            "bioclip_true_score": bc["bioclip_true_score"],
            "sam3_localised":     s3["sam3_localised"],
            "sam3_quality":       s3["sam3_quality"],
            "sam3_backend":       s3["sam3_backend"],
            "combined_correct":   combined_correct,
        })
    return pd.DataFrame(rows)


def print_report(
    frame: pd.DataFrame,
    target_species: list[str],
    bioclip_elapsed: float,
    sam3_elapsed: float,
) -> str:
    total = len(frame)
    bc_correct = int(frame["bioclip_correct"].sum())
    s3_localised = int(frame["sam3_localised"].sum())
    combined = int(frame["combined_correct"].sum())

    lines = [
        "=" * 70,
        "BioCLIP vs SAM 3 — Comparison Report",
        "=" * 70,
        "",
        f"{'Metric':<40} {'BioCLIP':>10} {'SAM 3':>8} {'Combined':>10}",
        "-" * 70,
        f"{'Images evaluated':<40} {total:>10} {total:>8} {total:>10}",
        f"{'Correct / Localised':<40} {bc_correct:>10} {s3_localised:>8} {combined:>10}",
        f"{'Accuracy / Localisation rate':<40} "
        f"{bc_correct/total*100:>9.1f}% "
        f"{s3_localised/total*100:>7.1f}% "
        f"{combined/total*100:>9.1f}%",
        f"{'Total runtime':<40} {bioclip_elapsed:>9.1f}s "
        f"{sam3_elapsed:>7.1f}s {'—':>10}",
        f"{'Avg time per image':<40} "
        f"{bioclip_elapsed/total:>9.2f}s "
        f"{sam3_elapsed/total:>7.2f}s {'—':>10}",
        "",
        "What these numbers mean:",
        "  BioCLIP accuracy   = % images where top-1 species prediction is correct",
        "  SAM 3 localisation = % images where SAM 3 found a valid mask",
        "                       (given the correct species name as prompt)",
        "  Combined           = both BioCLIP identified AND SAM 3 localised",
        "",
        f"{'Species':<22} {'BioCLIP':>10} {'SAM 3':>8} {'Combined':>10} {'N':>4}",
        "-" * 70,
    ]

    for sp in target_species:
        sp_frame = frame[frame["true_species"] == sp]
        if sp_frame.empty:
            continue
        n = len(sp_frame)
        bc_acc = sp_frame["bioclip_correct"].mean() * 100
        s3_acc = sp_frame["sam3_localised"].mean() * 100
        cb_acc = sp_frame["combined_correct"].mean() * 100
        winner = "BioCLIP" if bc_acc > s3_acc else "SAM 3"
        lines.append(
            f"{sp:<22} {bc_acc:>9.1f}% {s3_acc:>7.1f}% "
            f"{cb_acc:>9.1f}% {n:>4}"
        )

    lines += [
        "",
        "Top BioCLIP misclassifications:",
        "-" * 70,
    ]
    wrong = frame[~frame["bioclip_correct"]]
    if not wrong.empty:
        confusions = (wrong.groupby(["true_species", "bioclip_predicted"])
                      .size().reset_index(name="count")
                      .sort_values("count", ascending=False).head(8))
        for _, row in confusions.iterrows():
            lines.append(
                f"  {row["true_species"]:<18} -> {row["bioclip_predicted"]:<18} "
                f"({int(row['count'])} times)"
            )
    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(species_filter: list[str] | None = None, top_n: int | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SUBSET_CSV.exists():
        print("[ERROR] subset_metadata.csv not found. Run setup_data.py first.")
        return

    full_subset = pd.read_csv(SUBSET_CSV)
    available = list(full_subset["species_label"].unique())
    target = [s for s in (species_filter or TARGET_SPECIES) if s in available]
    if not target:
        print(f"[ERROR] No matching species. Available: {sorted(available)}")
        return

    # Build evaluation frame
    frames = [full_subset[full_subset["species_label"] == sp].head(top_n or 9999)
              for sp in target]
    eval_frame = pd.concat(frames, ignore_index=True).copy()
    eval_frame["local_image_path"] = (
        eval_frame["local_image_path"].astype(str).apply(_resolve_image_path)
    )
    # Keep only images that exist on disk
    exists_mask = eval_frame["local_image_path"].apply(lambda p: Path(p).exists())
    eval_frame = eval_frame[exists_mask].reset_index(drop=True)
    print(f"\n[INFO] Evaluating {len(eval_frame)} images across {len(target)} species.")

    # ── BioCLIP evaluation ────────────────────────────────────────────────────
    bioclip = BioCLIPModel()
    bioclip_results, bioclip_elapsed = run_bioclip_evaluation(
        eval_frame, target, bioclip
    )
    bioclip.release()

    # ── SAM 3 evaluation ──────────────────────────────────────────────────────
    sam3_results, sam3_elapsed = run_sam3_evaluation(eval_frame, bioclip_results)

    # ── Merge and save ────────────────────────────────────────────────────────
    combined = build_combined_results(bioclip_results, sam3_results)

    out_csv = OUTPUT_DIR / "bioclip_vs_sam3_results.csv"
    combined.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {len(combined)} rows -> {out_csv}")

    report = print_report(combined, target, bioclip_elapsed, sam3_elapsed)
    print("\n" + report)

    summary_path = OUTPUT_DIR / "bioclip_vs_sam3_summary.txt"
    summary_path.write_text(report, encoding="utf-8")
    print(f"[INFO] Summary saved -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BioCLIP identification accuracy vs SAM 3 localisation rate"
    )
    parser.add_argument(
        "--species", nargs="+", default=None,
        help="Species to evaluate (default: all). e.g. --species zebra buffalo"
    )
    parser.add_argument(
        "--top_n", type=int, default=None,
        help="Max images per species (default: all). Use --top_n 15 for a quick run."
    )
    args = parser.parse_args()
    run(species_filter=args.species, top_n=args.top_n)
