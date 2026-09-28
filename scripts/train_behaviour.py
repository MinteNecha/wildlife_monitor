"""
Train a behavioural model (Pipeline 2) and save it for the dashboard to serve.

Replaces the earlier run_training.py / run_training_transformer.py pair: one
script, with the architecture chosen by a flag. Every run writes a checkpoint
under models/behaviour/ and a metrics file under results/behaviour/, so the
dashboard can load the model and show how it scored.

Usage:
    python scripts/train_behaviour.py --species zebra
    python scripts/train_behaviour.py --species zebra --model transformer
    python scripts/train_behaviour.py --all --model lstm
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
import numpy as np

from wildlife_monitor.config import RESULTS_DIR
from wildlife_monitor.db import database_exists
from wildlife_monitor.pipeline2.datasets import (
    DEFAULT_EPOCHS, DEFAULT_PIPELINE, METRICS_DIR, load_species_detections,
)
from wildlife_monitor.pipeline2.inference import BEHAVIOUR_DIR, checkpoint_path
from wildlife_monitor.pipeline2.models import build_model
from wildlife_monitor.pipeline2.train import (
    build_training_set, split_train_test, examples_to_tensors,
    train_model, evaluate_model,
)




def _store_patterns(species: str, architecture: str, frame) -> int:
    """Run inference with the just-saved model and persist the patterns."""
    from wildlife_monitor.db import save_patterns
    from wildlife_monitor.pipeline2.inference import BehaviourService
    try:
        service = BehaviourService.load(species, architecture)
        patterns = service.predict(frame, species)
        return save_patterns(patterns, service.model_version)
    except Exception as error:
        print(f"[WARN] Could not store patterns: {error}")
        return 0


def train_one(species: str, architecture: str, pipeline: str, max_length: int,
              epochs: int | None, learning_rate: float, seed: int,
              batch_size: int | None = None) -> dict:
    frame, source = load_species_detections(species, pipeline)
    np.random.seed(seed)
    epochs = epochs or DEFAULT_EPOCHS.get(architecture, 100)

    print(f"[INFO] {len(frame):,} detections for {species} (source: {source})")
    examples = build_training_set(frame, max_length=max_length, species=species)
    train_examples, test_examples = split_train_test(examples)

    (train_sequences, train_lengths, train_activity,
     train_movement, train_months) = examples_to_tensors(train_examples)
    (test_sequences, test_lengths, test_activity,
     test_movement, test_months) = examples_to_tensors(test_examples)

    overrides = {"max_length": max_length} if architecture == "transformer" else {}
    if architecture == "transformer":
        overrides["num_layers"] = 1
    model = build_model(architecture, **overrides)

    print(f"Training {type(model).__name__} on {len(train_examples)} real "
          f"{species} camera sequences...")
    train_model(model, train_sequences, train_lengths, train_activity,
                train_movement, train_months, num_epochs=epochs,
                learning_rate=learning_rate, batch_size=batch_size)

    print("\n" + "=" * 60)
    print(f"Evaluating on {len(test_examples)} held-out cameras "
          f"(never seen during training):")
    print("=" * 60)
    metrics = evaluate_model(model, test_sequences, test_lengths, test_activity,
                             test_movement, test_months)

    label_counts = {}
    for example in examples:
        label_counts[example["movement_label"]] = (
            label_counts.get(example["movement_label"], 0) + 1)

    metadata = {
        "species": species,
        "architecture": architecture,
        "pipeline": pipeline,
        "source": source,
        "max_length": max_length,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "seed": seed,
        "batch_size": batch_size,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_cameras": len(train_examples),
        "test_cameras": len(test_examples),
        "movement_label_counts": label_counts,
        "metrics": metrics,
    }

    saved = model.save_checkpoint(checkpoint_path(species, architecture), metadata)

    # Classify every camera with the freshly trained model and store the
    # patterns, so the query and validation surfaces have something to work
    # against without re-running inference.
    stored = _store_patterns(species, architecture, frame)
    metadata["patterns_stored"] = stored
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{species}_{architecture}_metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2))

    print(f"\n[INFO] Checkpoint -> {saved}")
    print(f"[INFO] Patterns   -> {stored} camera classifications stored")
    print(f"[INFO] Metrics    -> {metrics_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="e.g. buffalo, zebra, wildebeest")
    group.add_argument("--all", action="store_true",
                        help="train every species with a detections CSV")
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=None,
                        help="defaults to 100 (LSTM) or 40 (Transformer)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int,
                        help="train on this many camera sequences at a time. "
                             "Omit for full-batch, which is the default and "
                             "reproduces earlier results exactly. Set it to "
                             "match a cross-species run when comparing the "
                             "two, since mini-batching changes how many "
                             "weight updates each epoch performs")
    args = parser.parse_args()

    BEHAVIOUR_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        species_list = []
        if database_exists():
            from wildlife_monitor.db import session, DetectionRepository
            with session() as connection:
                species_list = DetectionRepository(
                    connection).species_with_detections(args.pipeline)
        if not species_list:
            directory = RESULTS_DIR / args.pipeline
            species_list = sorted(path.stem.replace("detections_", "")
                                   for path in directory.glob("detections_*.csv"))
        if not species_list:
            print("[ERROR] No detections found in the database or on disk.")
            return
    else:
        species_list = [args.species]

    summaries = []
    for species in species_list:
        print("\n" + "#" * 60)
        print(f"# {species} · {args.model}")
        print("#" * 60)
        try:
            summaries.append(train_one(species, args.model, args.pipeline,
                                        args.max_length, args.epochs,
                                        args.lr, args.seed, args.batch_size))
        except FileNotFoundError as error:
            print(f"[SKIP] {error}")

    if len(summaries) > 1:
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'species':<18}{'activity':>10}{'movement':>10}")
        for summary in summaries:
            metrics = summary["metrics"]
            print(f"{summary['species']:<18}"
                  f"{metrics['activity_accuracy']:>9.1f}%"
                  f"{metrics['movement_accuracy']:>9.1f}%")


if __name__ == "__main__":
    main()
