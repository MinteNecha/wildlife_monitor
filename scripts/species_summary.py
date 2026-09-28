"""
Roll the per-camera classifications up into one statement per species (VL2).

Why this exists
---------------
Every accuracy figure this project produces measures agreement with its own
quartile rule. None of them check the system against real animals.

Published ecology is written about species, not cameras. "Lions are
nocturnal" is the kind of claim VL2 asks the results to be compared against.
This script counts each species' camera classifications into a single label so
that comparison can be made.

It stops there on purpose. It does not decide whether the answer matches the
literature, because which source is authoritative for a species is a judgement
for the ecologist. A system that graded itself against a hard-coded table
would look like validation without being it.

Reading the output
------------------
Each species gets a label and the share of its cameras that voted for it.
A high share means a real pattern. Where no label is clearly ahead the result
is marked ``split`` rather than given a winner, because a 40 per cent majority
across three classes is not a finding.

The rule labels are shown alongside the model predictions. If a species
disagrees with published ecology, that column says whether to question the
model or the labelling rule behind it.

Usage:
    python scripts/species_summary.py --all
    python scripts/species_summary.py --species zebra,wildebeest
    python scripts/species_summary.py --all --model transformer
    python scripts/species_summary.py --all --full      # every class count
"""

from __future__ import annotations

import argparse

import pandas as pd

from wildlife_monitor.config import TARGET_SPECIES
from wildlife_monitor.pipeline2.aggregate import (
    MIN_DETECTIONS, counts_frame, profile_species, to_frame,
)
from wildlife_monitor.pipeline2.datasets import (
    DEFAULT_PIPELINE, METRICS_DIR, load_species_detections,
)
from wildlife_monitor.pipeline2.inference import BehaviourService


def profile_one(species: str, architecture: str, pipeline: str,
                min_detections: int):
    """Classify one species' cameras, then count the votes."""
    try:
        frame, _ = load_species_detections(species, pipeline)
    except FileNotFoundError:
        return None, "no detections"

    try:
        service = BehaviourService.load_for(species, architecture)
    except (FileNotFoundError, OSError):
        return None, "no trained model"

    predictions = service.predict_frame(frame, species)
    if predictions.empty:
        return None, "no cameras classified"

    profile = profile_species(predictions, species, min_detections)
    return profile, ("cross-species model" if service.is_cross_species
                     else "own model")


def print_table(profiles, sources: dict[str, str]) -> pd.DataFrame:
    """The headline table: one row per species."""
    frame = to_frame(profiles)

    print("\n" + "=" * 96)
    print("SPECIES-LEVEL BEHAVIOUR, COUNTED FROM CAMERA CLASSIFICATIONS")
    print("=" * 96)
    print(f"{'Species':<20}{'Cams':>6}{'Activity':>14}{'Agree':>8}"
          f"{'Movement':>14}{'Agree':>8}{'Group size':>14}{'Agree':>8}")
    print("-" * 96)

    for profile in profiles:
        activity = profile.activity.label if profile.activity.clear else "split"
        movement = profile.movement.label if profile.movement.clear else "split"
        print(f"{profile.species:<20}{profile.cameras_counted:>6}"
              f"{activity:>14}{profile.activity.percentage:>8}"
              f"{movement:>14}{profile.movement.percentage:>8}"
              f"{profile.social.label:>14}{profile.social.percentage:>8}")

    print("-" * 96)
    print(f"'Agree' is the share of that species' cameras voting for the "
          f"label. 'split' means no label was\nclearly ahead. Cameras with "
          f"fewer than {MIN_DETECTIONS} detections are not counted.")
    return frame


def print_model_versus_rule(profiles) -> None:
    """Where the model's species-level answer differs from the rule's."""
    rows = []
    for profile in profiles:
        agreement = profile.model_and_rule_agree
        if not agreement or all(agreement.values()):
            continue
        for dimension, agrees in agreement.items():
            if agrees:
                continue
            predicted = getattr(profile, dimension)
            derived = getattr(profile, f"rule_{dimension}")
            rows.append((profile.species, dimension, predicted.label,
                         derived.label))

    if not rows:
        print("\nThe model and the labelling rule give the same species-level "
              "answer everywhere.\nAny disagreement with published ecology "
              "therefore points at the rule, not the model.")
        return

    print("\n" + "=" * 72)
    print("WHERE THE MODEL AND THE LABELLING RULE DISAGREE")
    print("=" * 72)
    print(f"{'Species':<20}{'Dimension':<12}{'Model says':<16}{'Rule says':<16}")
    print("-" * 72)
    for species, dimension, predicted, derived in rows:
        print(f"{species:<20}{dimension:<12}{predicted:<16}{derived:<16}")
    print("-" * 72)
    print("For these, a mismatch with published ecology points at the model.\n"
          "Everywhere else it points at the labelling rule.")


def print_detail(profiles) -> None:
    """Every class's camera count, so a thin majority is visible."""
    print("\n" + "=" * 72)
    print("FULL VOTE PER SPECIES")
    print("=" * 72)
    for profile in profiles:
        print(f"\n{profile.species}  "
              f"({profile.cameras_counted} cameras counted, "
              f"{profile.detections:,} detections)")
        for rollup in (profile.activity, profile.movement, profile.social):
            parts = " · ".join(
                f"{name} {count}" for name, count in rollup.counts.items())
            print(f"    {rollup.dimension:<12} {parts}")
        if profile.cameras_excluded:
            print(f"    {profile.cameras_excluded} camera(s) excluded as too "
                  f"thin")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="comma-separated species")
    group.add_argument("--all", action="store_true",
                       help="every configured species that has detections")
    parser.add_argument("--model", default="lstm",
                        choices=["lstm", "transformer"])
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--min-detections", type=int, default=MIN_DETECTIONS,
                        help=f"cameras below this are not counted "
                             f"(default {MIN_DETECTIONS})")
    parser.add_argument("--full", action="store_true",
                        help="also print every class count per species")
    parser.add_argument("--out", help="write the summary CSV here")
    args = parser.parse_args()

    requested = (TARGET_SPECIES if args.all
                 else [s.strip() for s in args.species.split(",") if s.strip()])

    profiles, sources, skipped = [], {}, []
    for species in requested:
        profile, source = profile_one(species, args.model, args.pipeline,
                                       args.min_detections)
        if profile is None:
            skipped.append(f"{species} ({source})")
            continue
        profiles.append(profile)
        sources[species] = source

    if skipped:
        print(f"[INFO] skipped: {', '.join(skipped)}")
    if not profiles:
        print("\n[ERROR] Nothing to summarise. Train a behaviour model first:")
        print("        python scripts/train_behaviour.py --all --model lstm")
        print("        python scripts/train_cross_species.py --all")
        raise SystemExit(1)

    pooled = [name for name, source in sources.items()
              if source == "cross-species model"]
    if pooled:
        print(f"[NOTE] classified by the cross-species model (never trained "
              f"on them): {', '.join(pooled)}")

    frame = print_table(profiles, sources)
    print_model_versus_rule(profiles)
    if args.full:
        print_detail(profiles)

    destination = args.out or (METRICS_DIR / f"species_summary_{args.model}.csv")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    print(f"\n[INFO] Summary  -> {destination}")

    detail_path = str(destination).replace(".csv", "_counts.csv")
    counts_frame(profiles).to_csv(detail_path, index=False)
    print(f"[INFO] Full vote -> {detail_path}")
    print("\nCompare the Activity and Movement columns against published "
          "ecology for each species.\nThat comparison is the part this script "
          "deliberately leaves to you.")


if __name__ == "__main__":
    main()
