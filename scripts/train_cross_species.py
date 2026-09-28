"""
Train one behavioural model across species, and test it on species it has
never seen (FR5, VL3).

Why this exists
---------------
``train_behaviour.py`` trains one model per species and tests it on held-out
*cameras* of that same species. The model therefore only ever sees one animal,
so a good score cannot distinguish two very different things: learning what a
migratory detection history looks like, or memorising the quirks of one
species' cameras.

This script trains a single model on several species at once and evaluates it
on a species excluded from training entirely. Anything it gets right there
came from the shape of the detection history, because it has never seen the
animal. That is the test FR5 asks for ("minimum classification accuracy of
75% on held-out species") and the one VL3 specifies ("train the model
excluding 15% of species entirely").

Leave-one-species-out
---------------------
With a handful of species a single held-out split depends heavily on which
species is chosen. By default every species takes a turn as the held-out set,
which gives one generalisation result per species rather than one overall.
It also makes each test set much larger than the per-species runs: a whole
species of cameras rather than 20 per cent of one.

The deployed model is trained on every species afterwards, because a model an
ecologist uses should see all the data available.

A note on labels
----------------
Labels are computed per species *before* pooling. The movement rule compares a
camera against ``quantile(0.75)`` of the other cameras, so pooling the frames
first would compute that quantile across a mixture of animals and relabel
every camera. Computing per species keeps every label identical to the
per-species runs, so the two sets of results are comparable.

Usage:
    python scripts/train_cross_species.py --all
    python scripts/train_cross_species.py --species buffalo,zebra,wildebeest
    python scripts/train_cross_species.py --all --model transformer
    python scripts/train_cross_species.py --all --holdout zebra
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from wildlife_monitor.config import TARGET_SPECIES
from wildlife_monitor.pipeline2.datasets import (
    DEFAULT_EPOCHS, DEFAULT_PIPELINE, METRICS_DIR, gather_species,
)
from wildlife_monitor.pipeline2.inference import (
    CROSS_SPECIES_KEY, checkpoint_path,
)
from wildlife_monitor.pipeline2.models import build_model
from wildlife_monitor.pipeline2.train import (
    MOVEMENT_CLASSES, build_cross_species_set, evaluate_model,
    examples_to_tensors, leave_one_species_out, species_in,
    split_by_species, train_model,
)

# Three movement classes, so a model that guesses would land near this.
CHANCE = 100.0 / len(MOVEMENT_CLASSES)


def fit(train_examples, architecture: str, max_length: int, epochs: int,
        learning_rate: float, batch_size: int = 32):
    """Build and train one model on the given examples."""
    sequences, lengths, activity, movement, months = examples_to_tensors(
        train_examples)

    overrides = {"max_length": max_length} if architecture == "transformer" else {}
    if architecture == "transformer":
        overrides["num_layers"] = 1
    model = build_model(architecture, **overrides)

    train_model(model, sequences, lengths, activity, movement, months,
                num_epochs=epochs, learning_rate=learning_rate, verbose=False,
                batch_size=batch_size)
    return model


def run_fold(held_out: str, train_examples, test_examples, architecture: str,
             max_length: int, epochs: int, learning_rate: float,
             seed: int, batch_size: int = 32) -> dict:
    """Train without one species, then evaluate on it."""
    np.random.seed(seed)
    print(f"\n  hold out {held_out:<18} "
          f"train {len(train_examples):>4} cameras "
          f"({', '.join(species_in(train_examples))})")

    model = fit(train_examples, architecture, max_length, epochs,
                learning_rate, batch_size)
    sequences, lengths, activity, movement, months = examples_to_tensors(
        test_examples)
    metrics = evaluate_model(model, sequences, lengths, activity, movement,
                             months, verbose=False)

    print(f"  {'':<27} test  {len(test_examples):>4} cameras  "
          f"activity {metrics['activity_accuracy']:>5.1f}%  "
          f"movement {metrics['movement_accuracy']:>5.1f}%")
    return {"held_out_species": held_out,
            "train_cameras": len(train_examples),
            "train_species": species_in(train_examples),
            "test_cameras": len(test_examples),
            **metrics}


def summarise(folds: list[dict]) -> None:
    """Print the per-fold table and say plainly what it means."""
    print("\n" + "=" * 72)
    print("GENERALISATION TO UNSEEN SPECIES")
    print("=" * 72)
    print(f"{'Held-out species':<20}{'Test cams':>10}{'Activity':>11}"
          f"{'Movement':>11}{'vs chance':>12}")
    print("-" * 72)
    for fold in folds:
        gap = fold["movement_accuracy"] - CHANCE
        print(f"{fold['held_out_species']:<20}{fold['test_cameras']:>10}"
              f"{fold['activity_accuracy']:>10.1f}%"
              f"{fold['movement_accuracy']:>10.1f}%"
              f"{gap:>+11.1f}")

    movement = [f["movement_accuracy"] for f in folds]
    activity = [f["activity_accuracy"] for f in folds]
    mean_movement = sum(movement) / len(movement)
    print("-" * 72)
    print(f"{'mean':<20}{'':>10}{sum(activity)/len(activity):>10.1f}%"
          f"{mean_movement:>10.1f}%{mean_movement - CHANCE:>+11.1f}")
    print(f"{'range':<20}{'':>10}{'':>11}"
          f"{min(movement):>6.1f} to {max(movement):<5.1f}")

    print()
    if mean_movement >= 70:
        print("Reading: movement classification transfers to species the model")
        print("has never seen. The model learned the shape of a detection")
        print("history rather than the identity of one animal.")
    elif mean_movement >= CHANCE + 10:
        print("Reading: movement classification partly transfers. The model")
        print("carries some species-independent signal, but well short of its")
        print("per-species accuracy. Compare the folds below for which species")
        print("transfer and which do not.")
    else:
        print(f"Reading: movement classification does not transfer. At "
              f"{mean_movement:.1f}% against a {CHANCE:.1f}% chance baseline,")
        print("the per-species models were learning species-specific patterns")
        print("rather than behaviour. This is a negative result, and it is the")
        print("honest answer to the generalisation question.")

    spread = max(movement) - min(movement)
    if spread >= 20:
        best = max(folds, key=lambda f: f["movement_accuracy"])
        worst = min(folds, key=lambda f: f["movement_accuracy"])
        print(f"\nThe folds disagree by {spread:.1f} points. "
              f"{best['held_out_species']} transfers best "
              f"({best['movement_accuracy']:.1f}%) and "
              f"{worst['held_out_species']} worst "
              f"({worst['movement_accuracy']:.1f}%).")
        print("A species whose ecology differs from the training set is the "
              "harder case, and that variation is itself a result.")

    print("\nCaveat: these labels come from a quantile rule computed within "
          "each\nspecies, so this measures whether the rule's shape transfers, "
          "not\nwhether real animal behaviour does.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="comma-separated species to pool")
    group.add_argument("--all", action="store_true",
                       help="pool every configured species that has detections")
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--holdout",
                        help="evaluate one named species only, instead of "
                             "rotating through every species")
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="camera sequences trained on at once. Lower this "
                             "if the run exhausts memory; raise it if you have "
                             "room to spare (default 64)")
    parser.add_argument("--no-deploy", action="store_true",
                        help="report generalisation only; do not train or save "
                             "the final all-species model")
    args = parser.parse_args()

    requested = (TARGET_SPECIES if args.all
                 else [s.strip() for s in args.species.split(",") if s.strip()])
    epochs = args.epochs or DEFAULT_EPOCHS.get(args.model, 100)

    sources = gather_species(requested, args.pipeline)
    if len(sources) < 2:
        print(f"\n[ERROR] Cross-species training needs at least two species "
              f"with detections. Found {len(sources)}.")
        print("        Run Pipeline 1 for more species, or import existing "
              "results with 'python scripts/import_results.py'.")
        raise SystemExit(1)

    print(f"\n[INFO] Building pooled training set "
          f"({len(sources)} species, labels computed per species)")
    examples = build_cross_species_set(sources, max_length=args.max_length)
    print(f"[INFO] {len(examples)} camera sequences across "
          f"{', '.join(species_in(examples))}")

    counts: dict[str, int] = {}
    for example in examples:
        counts[example["movement_label"]] = counts.get(
            example["movement_label"], 0) + 1
    print("[INFO] movement labels: " + ", ".join(
        f"{name} {counts.get(name, 0)}" for name in MOVEMENT_CLASSES))

    steps = max(1, -(-len(examples) // args.batch_size))
    print(f"\n[INFO] Leave-one-species-out ({args.model}, {epochs} epochs, "
          f"batch {args.batch_size}, about {steps} updates per epoch)")
    folds = []
    if args.holdout:
        train_examples, test_examples = split_by_species(examples, args.holdout)
        if not test_examples:
            print(f"[ERROR] No examples for held-out species "
                  f"'{args.holdout}'. Available: "
                  f"{', '.join(species_in(examples))}")
            raise SystemExit(1)
        folds.append(run_fold(args.holdout, train_examples, test_examples,
                              args.model, args.max_length, epochs,
                              args.learning_rate, args.seed,
                              args.batch_size))
    else:
        for held_out, train_examples, test_examples in leave_one_species_out(
                examples):
            folds.append(run_fold(held_out, train_examples, test_examples,
                                  args.model, args.max_length, epochs,
                                  args.learning_rate, args.seed,
                                  args.batch_size))

    summarise(folds)

    movement = [f["movement_accuracy"] for f in folds]
    report = {
        "architecture": args.model,
        "pipeline": args.pipeline,
        "species": species_in(examples),
        "total_cameras": len(examples),
        "max_length": args.max_length,
        "epochs": epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "chance_baseline": round(CHANCE, 2),
        "movement_label_counts": counts,
        "folds": folds,
        "mean_movement_accuracy": round(sum(movement) / len(movement), 2),
        "mean_activity_accuracy": round(
            sum(f["activity_accuracy"] for f in folds) / len(folds), 2),
    }

    if not args.no_deploy:
        print(f"\n[INFO] Training the deployed model on all "
              f"{len(species_in(examples))} species")
        np.random.seed(args.seed)
        model = fit(examples, args.model, args.max_length, epochs,
                    args.learning_rate, args.batch_size)
        metadata = {
            "species": CROSS_SPECIES_KEY,
            "trained_species": species_in(examples),
            "architecture": args.model,
            "cross_species": True,
            "train_cameras": len(examples),
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "generalisation": report,
            **{key: report[key] for key in
               ("pipeline", "max_length", "epochs", "learning_rate", "seed",
                "batch_size")},
        }
        saved = model.save_checkpoint(
            checkpoint_path(CROSS_SPECIES_KEY, args.model), metadata)
        print(f"[INFO] Checkpoint -> {saved}")
        print("[INFO] The dashboard will now use this model for any species "
              "without its own trained model.")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METRICS_DIR / f"cross_species_{args.model}_metrics.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[INFO] Metrics    -> {report_path}")


if __name__ == "__main__":
    main()
