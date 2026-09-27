"""
Training one behavioural model across species, and testing it on unseen ones.

Three properties matter most here, and each has a test that fails loudly if it
breaks.

**Labels must survive pooling.** The movement rule compares a camera against
``quantile(0.75)`` of the other cameras. Pooling several species before
computing that quantile would silently relabel every camera and make the
cross-species results incomparable with the per-species ones. Nothing would
crash, so only a test catches it.

**The held-out species must not leak into training.** If it does, the
generalisation figure measures nothing and reads as a success.

**An untrained species must still get an answer, clearly marked.** The whole
point of the pooled model is the ecologist whose species was never trained. A
prediction that does not say it came from a model which has never seen the
animal is worse than no prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildlife_monitor.pipeline2 import inference as inf
from wildlife_monitor.pipeline2.models import build_model
from wildlife_monitor.pipeline2.train import (
    MOVEMENT_CLASSES, build_cross_species_set, build_training_set,
    evaluate_model, examples_to_tensors, leave_one_species_out, species_in,
    split_by_species, split_train_test, train_model,
)

SPECIES = ["buffalo", "zebra", "wildebeest"]
CHANCE = 100.0 / len(MOVEMENT_CLASSES)


def detections(species: str, seed: int = 0, cameras: int = 15) -> pd.DataFrame:
    """Cameras with three deliberately different seasonal shapes.

    Every species uses the same three shapes, so a model that learns shape
    rather than species should transfer. That is the property under test, and
    the negative control below confirms the harness can still fail.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for camera in range(cameras):
        kind = camera % 3
        if kind == 0:                                  # one month only
            months = [3] * 30
        elif kind == 1:                                # every month, many
            months = list(np.repeat(np.arange(1, 13), 3))
        else:                                          # thin and scattered
            months = [5, 9, 11]
        hour = 9 if camera % 2 else 21
        for index, month in enumerate(months):
            rows.append({
                "detection_id": f"{species}-{camera}-{index}",
                "image_id": f"{species}_{camera}_{index}",
                "pipeline": "bioclip_megadetector",
                "timestamp": f"2010-{int(month):02d}-{index % 27 + 1:02d} "
                              f"{hour:02d}:30:00",
                "camera_id": f"{species.upper()}{camera:02d}",
                "latitude": -2.3, "longitude": 34.8,
                "habitat_type": "open_grassland",
                "species": species, "confidence": 0.9,
                "location_type": "bbox", "location": "1,2,3,4",
                "detection_quality": 0.8,
                "instance_count": int(rng.integers(1, 6)),
                "image_path": f"images/{species}_{camera}.jpg",
                "mask_path": "", "overlay_path": "",
                "ground_truth_species": species, "correct": "correct",
            })
    return pd.DataFrame(rows)


@pytest.fixture
def sources() -> dict[str, pd.DataFrame]:
    return {name: detections(name, seed=index)
            for index, name in enumerate(SPECIES)}


@pytest.fixture
def pooled(sources) -> list[dict]:
    return build_cross_species_set(sources)


def fit(examples, epochs: int = 40):
    np.random.seed(42)
    sequences, lengths, activity, movement, months = examples_to_tensors(
        examples)
    model = build_model("lstm")
    train_model(model, sequences, lengths, activity, movement, months,
                num_epochs=epochs, verbose=False)
    return model


def score(model, examples) -> float:
    sequences, lengths, activity, movement, months = examples_to_tensors(
        examples)
    return evaluate_model(model, sequences, lengths, activity, movement,
                           months, verbose=False)["movement_accuracy"]


# ── pooling ──────────────────────────────────────────────────────────────────

def test_every_pooled_example_carries_its_species(pooled):
    assert species_in(pooled) == sorted(SPECIES)
    assert all(example["species"] in SPECIES for example in pooled)


def test_pooling_does_not_change_a_single_label(sources, pooled):
    """The property that makes cross-species results comparable.

    Labels are computed per species and only then pooled. Were they computed
    after pooling, the 75th percentile would span a mixture of animals and
    every label could move.
    """
    alone = {}
    for species, frame in sources.items():
        for example in build_training_set(frame, species=species):
            alone[(species, example["camera_id"])] = (
                example["activity_label"], example["movement_label"])

    assert len(alone) == len(pooled)
    for example in pooled:
        expected = alone[(example["species"], example["camera_id"])]
        assert (example["activity_label"], example["movement_label"]) == \
            expected, f"pooling changed the label for {example['camera_id']}"


def test_pooled_set_is_the_sum_of_its_species(sources, pooled):
    per_species = sum(len(build_training_set(frame, species=name))
                      for name, frame in sources.items())
    assert len(pooled) == per_species


# ── splitting ────────────────────────────────────────────────────────────────

def test_holding_out_a_species_removes_it_from_training(pooled):
    train, test = split_by_species(pooled, "zebra")

    assert species_in(test) == ["zebra"]
    assert "zebra" not in species_in(train)
    assert len(train) + len(test) == len(pooled)


def test_the_held_out_species_cannot_leak_through_a_camera(pooled):
    """No camera may appear on both sides of a species split."""
    train, test = split_by_species(pooled, "zebra")
    train_cameras = {example["camera_id"] for example in train}
    test_cameras = {example["camera_id"] for example in test}
    assert not (train_cameras & test_cameras)


def test_several_species_can_be_held_out_at_once(pooled):
    train, test = split_by_species(pooled, ["zebra", "buffalo"])
    assert species_in(test) == ["buffalo", "zebra"]
    assert species_in(train) == ["wildebeest"]


def test_species_names_are_matched_without_case_or_spacing(pooled):
    _, test = split_by_species(pooled, "  ZEBRA ")
    assert species_in(test) == ["zebra"]


def test_holding_out_an_absent_species_trains_on_everything(pooled):
    train, test = split_by_species(pooled, "impala")
    assert len(train) == len(pooled)
    assert test == []


def test_leave_one_species_out_covers_every_species_exactly_once(pooled):
    folds = list(leave_one_species_out(pooled))

    assert [held for held, _, _ in folds] == sorted(SPECIES)
    for held, train, test in folds:
        assert species_in(test) == [held]
        assert held not in species_in(train)
        assert len(train) + len(test) == len(pooled)


def test_a_species_split_is_not_a_camera_split(pooled):
    """The two splits answer different questions and must not be confused."""
    _, species_test = split_by_species(pooled, "zebra")
    _, camera_test = split_train_test(pooled)

    assert species_in(species_test) == ["zebra"]
    # A camera split leaves every species represented on both sides.
    assert len(species_in(camera_test)) > 1


# ── the experiment itself ────────────────────────────────────────────────────

def test_a_model_transfers_to_a_species_it_never_trained_on(pooled):
    """On data where the same shape means the same label, transfer happens."""
    train, test = split_by_species(pooled, "zebra")
    accuracy = score(fit(train), test)
    assert accuracy > CHANCE + 20, (
        f"expected transfer on consistent data, got {accuracy:.1f}%")


def test_shuffled_labels_collapse_to_chance(pooled):
    """The negative control.

    If a model scores well against permuted labels, the evaluation is
    measuring an artefact rather than signal, and every figure the experiment
    produces would be meaningless.
    """
    train, test = split_by_species(pooled, "zebra")
    model = fit(train)

    sequences, lengths, activity, movement, months = examples_to_tensors(test)
    shuffled = np.random.default_rng(0).permutation(movement)
    accuracy = evaluate_model(model, sequences, lengths, activity, shuffled,
                               months, verbose=False)["movement_accuracy"]

    assert accuracy < CHANCE + 20, (
        f"shuffled labels scored {accuracy:.1f}%, which means the evaluation "
        f"is not measuring real signal")


def test_every_held_out_camera_receives_a_prediction(pooled):
    train, test = split_by_species(pooled, "zebra")
    model = fit(train, epochs=5)
    sequences, lengths, activity, movement, months = examples_to_tensors(test)
    metrics = evaluate_model(model, sequences, lengths, activity, movement,
                             months, verbose=False)
    assert metrics["test_cameras"] == len(test)


# ── serving an untrained species ─────────────────────────────────────────────

@pytest.fixture
def trained_checkpoints(tmp_path, monkeypatch, pooled):
    """A cross-species model on disk, plus an own model for zebra only."""
    monkeypatch.setattr(inf, "BEHAVIOUR_DIR", tmp_path)

    pooled_model = fit(pooled, epochs=5)
    pooled_model.save_checkpoint(
        inf.checkpoint_path(inf.CROSS_SPECIES_KEY, "lstm"),
        {"species": inf.CROSS_SPECIES_KEY, "trained_species": SPECIES,
         "architecture": "lstm", "cross_species": True, "max_length": 40,
         "trained_at": "2026-09-27T22:00:00"})

    own = fit(pooled, epochs=5)
    own.save_checkpoint(
        inf.checkpoint_path("zebra", "lstm"),
        {"species": "zebra", "architecture": "lstm", "max_length": 40,
         "trained_at": "2026-09-27T22:00:00"})
    return tmp_path


def test_a_species_with_its_own_model_uses_it(trained_checkpoints):
    service = inf.BehaviourService.load_for("zebra", "lstm")
    assert not service.is_cross_species
    assert service.trained_species == ["zebra"]


def test_an_untrained_species_falls_back_to_the_pooled_model(
        trained_checkpoints):
    service = inf.BehaviourService.load_for("impala", "lstm")
    assert service.is_cross_species
    assert not service.saw_species("impala")
    assert service.trained_species == SPECIES


def test_the_fallback_says_the_species_was_never_seen(trained_checkpoints):
    """A prediction for an unseen species must not read like any other."""
    service = inf.BehaviourService.load_for("impala", "lstm")
    message = service.provenance("impala")

    assert "never seen" in message
    assert "impala" in message
    assert "lower accuracy" in message


def test_a_trained_species_is_not_warned_about(trained_checkpoints):
    service = inf.BehaviourService.load_for("impala", "lstm")
    assert "never seen" not in service.provenance("buffalo")


def test_the_fallback_actually_classifies_the_unseen_species(
        trained_checkpoints):
    service = inf.BehaviourService.load_for("impala", "lstm")
    patterns = service.predict(detections("impala", seed=9), "impala")

    assert len(patterns) == 15
    assert all(pattern.movement_class in MOVEMENT_CLASSES
               for pattern in patterns)


def test_without_any_pooled_model_an_untrained_species_still_fails(
        tmp_path, monkeypatch):
    """The fallback must not invent a model that was never trained."""
    monkeypatch.setattr(inf, "BEHAVIOUR_DIR", tmp_path)
    with pytest.raises((FileNotFoundError, OSError)):
        inf.BehaviourService.load_for("impala", "lstm")


def test_the_pooled_checkpoint_is_labelled_not_listed_as_an_animal(
        trained_checkpoints):
    entries = {entry["species"]: entry for entry in inf.available_checkpoints()}

    assert entries[inf.CROSS_SPECIES_KEY]["cross_species"] is True
    assert entries[inf.CROSS_SPECIES_KEY]["label"] == "all species (cross-species)"
    assert entries["zebra"]["cross_species"] is False
    assert inf.species_with_models() == ["zebra"]


def test_model_version_does_not_expose_the_internal_key(trained_checkpoints):
    service = inf.BehaviourService.load_for("impala", "lstm")
    assert inf.CROSS_SPECIES_KEY not in service.model_version
    assert "cross-species" in service.model_version
