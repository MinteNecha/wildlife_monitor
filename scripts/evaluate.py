"""
BioCLIP multi-species evaluation — shows correctness live.

This is exactly how the 43.7% top-1 accuracy figure was computed.

For every image in the subset, BioCLIP scores it against ALL 11 species
text prompts simultaneously. The species whose prompt scores highest is
the predicted species. Comparing that prediction against the ground-truth
label from the subset metadata gives a correct / incorrect result per image.

This produces an evaluation CSV that shows:
  - which images were correctly identified
  - which were wrong and what BioCLIP predicted instead
  - per-species accuracy breakdown
  - overall top-1 accuracy

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --top_n 50   (evaluate top 50 per species)
    python scripts/evaluate.py --species zebra  (one species only)

Output:
    results/evaluation/evaluation_results.csv
    results/evaluation/evaluation_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from wildlife_monitor.config import (
    SUBSET_CSV, RESULTS_DIR, PROMPT_TEMPLATE, TARGET_SPECIES, SystemConfig
)
from wildlife_monitor.data import load_species_subset
from wildlife_monitor.models import BioCLIPModel


OUTPUT_DIR = RESULTS_DIR / "evaluation"


def evaluate(species_filter: str | None = None, top_n: int | None = None) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load full subset for ground truth
    if not SUBSET_CSV.exists():
        print("[ERROR] subset_metadata.csv not found. Run setup_data.py first.")
        return
    full_subset = pd.read_csv(SUBSET_CSV)

    # Choose which species to evaluate
    target = [species_filter] if species_filter else TARGET_SPECIES
    target = [s for s in target if s in full_subset["species_label"].unique()]
    if not target:
        print(f"[ERROR] No matching species found. Available: "
              f"{sorted(full_subset['species_label'].unique())}")
        return

    # Build the images to evaluate
    frames = []
    for sp in target:
        sp_frame = full_subset[full_subset["species_label"] == sp].copy()
        if top_n:
            sp_frame = sp_frame.head(top_n)
        frames.append(sp_frame)
    eval_frame = pd.concat(frames, ignore_index=True)

    # Resolve image paths (same logic as loader)
    from wildlife_monitor.data.loader import _resolve_image_path
    eval_frame["local_image_path"] = (
        eval_frame["local_image_path"].astype(str).apply(_resolve_image_path)
    )
    # Filter to images that actually exist on disk
    exists_mask = eval_frame["local_image_path"].apply(lambda p: Path(p).exists())
    missing = (~exists_mask).sum()
    if missing:
        print(f"[WARN] {missing} images not found on disk — skipping.")
    eval_frame = eval_frame[exists_mask].reset_index(drop=True)

    print(f"[INFO] Evaluating {len(eval_frame)} images across "
          f"{len(target)} species.")
    print(f"[INFO] Species: {target}")

    # Load BioCLIP once
    bioclip = BioCLIPModel()

    # Compute text embeddings for ALL target species (done ONCE, before image loop)
    print("[INFO] Computing text embeddings for all species prompts...")
    text_feats = {}
    for sp in target:
        prompt = PROMPT_TEMPLATE.format(species=sp)
        tokens = bioclip.tokenizer([prompt]).to(bioclip.device)
        with torch.no_grad():
            feat = bioclip.model.encode_text(tokens)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        text_feats[sp] = feat
    print(f"[INFO] {len(text_feats)} species prompts encoded.")

    # Score each image against all species prompts
    results = []
    for _, row in tqdm(eval_frame.iterrows(), total=len(eval_frame),
                       desc="Evaluating"):
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

        # Score against every species prompt
        scores = {}
        for sp, text_feat in text_feats.items():
            scores[sp] = float((img_feat @ text_feat.T).squeeze())

        predicted_species = max(scores, key=scores.get)
        correct = predicted_species == true_species

        results.append({
            "image_id":           str(row["image_id"]),
            "image_path":         str(img_path),
            "true_species":       true_species,
            "predicted_species":  predicted_species,
            "correct":            correct,
            "top1_score":         round(scores[predicted_species], 4),
            "true_species_score": round(scores.get(true_species, 0.0), 4),
            "camera_id":          str(row.get("site_id", "")),
            "latitude":           float(row.get("latitude", 0.0)),
            "longitude":          float(row.get("longitude", 0.0)),
            **{f"score_{sp}": round(scores[sp], 4) for sp in target},
        })

    bioclip.release()

    # Save results
    results_frame = pd.DataFrame(results)
    out_csv = OUTPUT_DIR / "evaluation_results.csv"
    results_frame.to_csv(out_csv, index=False)
    print(f"\n[INFO] Saved {len(results_frame)} evaluation records -> {out_csv}")

    # Build summary
    _print_summary(results_frame, target, out_csv)


def _print_summary(frame: pd.DataFrame, target: list[str], out_csv: Path) -> None:
    total = len(frame)
    correct = int(frame["correct"].sum())
    overall_acc = correct / total * 100 if total else 0.0

    lines = [
        "=" * 62,
        "BioCLIP Multi-Species Evaluation — Top-1 Accuracy",
        "=" * 62,
        "",
        f"{'Total images evaluated:':<36} {total}",
        f"{'Correct (top-1):':<36} {correct}",
        f"{'Overall top-1 accuracy:':<36} {overall_acc:.1f}%",
        "",
        f"{'Species':<22} {'Images':>8} {'Correct':>9} {'Accuracy':>10}",
        "-" * 62,
    ]

    for sp in target:
        sp_frame = frame[frame["true_species"] == sp]
        if sp_frame.empty:
            continue
        n = len(sp_frame)
        c = int(sp_frame["correct"].sum())
        acc = c / n * 100
        flag = " ✓" if acc >= 50 else " ✗"
        lines.append(f"{sp:<22} {n:>8} {c:>9} {acc:>9.1f}%{flag}")

    lines += [
        "=" * 62,
        "",
        "Incorrect predictions (top misclassifications):",
        "-" * 62,
    ]
    wrong = frame[~frame["correct"]]
    if not wrong.empty:
        confusions = (wrong.groupby(["true_species", "predicted_species"])
                      .size().reset_index(name="count")
                      .sort_values("count", ascending=False).head(10))
        for _, row in confusions.iterrows():
            lines.append(
                f"  {row['true_species']:<18} → {row['predicted_species']:<18} "
                f"({int(row['count'])} times)"
            )
    else:
        lines.append("  (no incorrect predictions)")

    lines += ["", f"Full results: {out_csv}"]
    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = OUTPUT_DIR / "evaluation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"[INFO] Summary saved -> {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BioCLIP multi-species evaluation — computes top-1 accuracy"
    )
    parser.add_argument(
        "--species", type=str, default=None,
        help="Evaluate one species only (default: all species in subset)"
    )
    parser.add_argument(
        "--top_n", type=int, default=None,
        help="Max images per species to evaluate (default: all)"
    )
    args = parser.parse_args()
    evaluate(species_filter=args.species, top_n=args.top_n)
