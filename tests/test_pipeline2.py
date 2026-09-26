"""
Feature extraction, sequence building, labelling, checkpointing and inference.

The labelling tests matter most: the movement rule is what every behavioural
result is measured against, and an earlier version of it could never produce
the migratory class at all. These tests pin that behaviour down.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from wildlife_monitor.pipeline2 import labelling
from wildlife_monitor.pipeline2.feature_extractor import (
    TemporalFeatureExtractor, compute_intervals, encode_cyclical,
    encode_habitat, extract_hour, extract_day_of_year,
    NUM_FEATURES, LEGACY_NUM_FEATURES, HABITAT_CLASSES,
)
from wildlife_monitor.pipeline2.inference import BehaviourService
from wildlife_monitor.pipeline2.models import (
    LSTMBehaviourModel, TransformerBehaviourModel, build_model, load_checkpoint,
)
from wildlife_monitor.pipeline2.sequence_builder import (
    SequenceBuilder, sequences_to_arrays,
)
from wildlife_monitor.pipeline2.train import (
    build_training_set, split_train_test, examples_to_tensors,
    train_model, evaluate_model, confusion_matrix, per_class_recall,
)


# ── Feature extraction ───────────────────────────────────────────────────────

def test_cyclical_encoding_wraps_around():
    """23:00 and 00:00 must be neighbours, not opposites."""
    late = encode_cyclical(23.0, 24.0)
    midnight = encode_cyclical(0.0, 24.0)
    midday = encode_cyclical(12.0, 24.0)
    distance = lambda a, b: math.dist(a, b)
    assert distance(late, midnight) < distance(late, midday)


def test_unparseable_timestamps_fall_back_rather_than_raise():
    assert extract_hour("not a date") == 12.0
    assert extract_day_of_year("") == 182.0


def test_compute_intervals_measures_gaps_in_hours():
    intervals = compute_intervals([
        "2010-01-01 00:00:00", "2010-01-01 06:00:00", "2010-01-02 06:00:00"])
    assert intervals[0] == 0.0          # nothing precedes the first detection
    assert intervals[1] == pytest.approx(6.0)
    assert intervals[2] == pytest.approx(24.0)


def test_habitat_encoding_is_one_hot_with_unknown_fallback():
    encoded = encode_habitat("woodland")
    assert sum(encoded) == 1
    assert len(encoded) == len(HABITAT_CLASSES)
    assert encode_habitat("somewhere else") == encode_habitat("unknown")


def test_extractor_widths_match_their_declared_constants():
    assert TemporalFeatureExtractor().num_features == NUM_FEATURES
    assert TemporalFeatureExtractor(legacy=True).num_features == LEGACY_NUM_FEATURES


def test_extractor_produces_one_vector_per_detection(detections):
    camera = detections[detections.camera_id == "THIN"]
    vectors = TemporalFeatureExtractor().extract(camera)
    assert len(vectors) == len(camera)
    assert all(len(vector) == NUM_FEATURES for vector in vectors)


# ── Sequence building ────────────────────────────────────────────────────────

def test_short_sequences_are_padded_and_report_their_real_length(detections):
    builder = SequenceBuilder(max_length=40)
    thin = detections[detections.camera_id == "THIN"]
    sequence = builder.build(thin, "THIN")

    assert len(sequence.vectors) == 40           # padded to the window
    assert sequence.real_length == 4             # but only 4 are real
    assert sequence.detection_count == 4
    assert not sequence.truncated
    assert all(value == 0.0 for value in sequence.vectors[-1])


def test_long_sequences_are_truncated_but_keep_their_true_count(detections):
    builder = SequenceBuilder(max_length=40)
    high = detections[detections.camera_id == "HIGH"]
    sequence = builder.build(high, "HIGH")

    assert sequence.real_length == 40
    assert sequence.detection_count == 120       # full history is remembered
    assert sequence.truncated


def test_builder_matches_a_models_input_width():
    legacy = SequenceBuilder.for_model(40, LEGACY_NUM_FEATURES)
    modern = SequenceBuilder.for_model(40, NUM_FEATURES)
    assert legacy.num_features == LEGACY_NUM_FEATURES
    assert modern.num_features == NUM_FEATURES


def test_month_features_capture_the_seasonal_peak(detections):
    sequences = {s.camera_id: s for s in
                 SequenceBuilder(40).build_all(detections, "zebra")}
    peak = sequences["PEAK"].month_features
    assert peak[2] == pytest.approx(1.0)         # March, all detections
    assert peak.sum() == pytest.approx(1.0)

    spread = sequences["HIGH"].month_features
    assert spread.max() < 0.5                    # no single dominant month


# ── Labelling rules ──────────────────────────────────────────────────────────

def test_activity_reflects_time_of_day(detections):
    by_camera = {camera: frame for camera, frame in detections.groupby("camera_id")}
    assert labelling.classify_activity(by_camera["HIGH"]) == "diurnal"
    assert labelling.classify_activity(by_camera["PEAK"]) == "nocturnal"


def test_social_structure_follows_the_largest_group_seen():
    frame = lambda count: pd.DataFrame({"instance_count": [1, count]})
    assert labelling.classify_social_structure(frame(1)) == "solitary"
    assert labelling.classify_social_structure(frame(4)) == "small group"
    assert labelling.classify_social_structure(frame(30)) == "large herd"


def test_migratory_class_is_reachable(detections):
    """A regression test: the movement rule once could never return migratory."""
    fidelity = labelling.compute_site_fidelity(detections)
    concentration = labelling.compute_temporal_concentration(detections)
    counts = detections.groupby("camera_id").size()

    labels = {camera: labelling.classify_movement(
        camera, fidelity, concentration, counts) for camera in counts.index}

    assert labels["PEAK"] == "migratory", "a single-month camera must read as migratory"
    assert "territorial" in labels.values()


def test_a_camera_below_the_detection_floor_is_never_migratory(detections):
    fidelity = labelling.compute_site_fidelity(detections)
    concentration = labelling.compute_temporal_concentration(detections)
    counts = detections.groupby("camera_id").size()

    label = labelling.classify_movement(
        "PEAK", fidelity, concentration, counts,
        min_detections_for_migratory=10_000)
    assert label != "migratory"


def test_labels_use_the_full_history_not_the_truncated_window(detections):
    """Truncation governs model input only; labels must see everything."""
    concentration = labelling.compute_temporal_concentration(detections)
    truncated = detections.groupby("camera_id").head(5)
    assert concentration["HIGH"] != pytest.approx(
        labelling.compute_temporal_concentration(truncated)["HIGH"])


# ── Training, checkpointing, inference ───────────────────────────────────────

def test_training_set_has_one_example_per_camera(detections):
    examples = build_training_set(detections, max_length=40, species="zebra")
    assert len(examples) == detections.camera_id.nunique()
    assert {example["camera_id"] for example in examples} == {"HIGH", "PEAK", "THIN"}
    assert all(len(example["month_features"]) == 12 for example in examples)


def test_train_test_split_is_deterministic_and_disjoint(detections):
    examples = build_training_set(detections, species="zebra")
    train_a, test_a = split_train_test(examples, seed=7)
    train_b, test_b = split_train_test(examples, seed=7)

    assert [e["camera_id"] for e in test_a] == [e["camera_id"] for e in test_b]
    assert not ({e["camera_id"] for e in train_a}
                & {e["camera_id"] for e in test_a})


@pytest.mark.parametrize("architecture", ["lstm", "transformer"])
def test_checkpoint_round_trip_is_exact(tmp_path, detections, architecture):
    """A reloaded model must produce identical outputs, or serving is a lie."""
    examples = build_training_set(detections, max_length=10, species="zebra")
    sequences, lengths, _, _, months = examples_to_tensors(examples)

    overrides = {"max_length": 10, "num_layers": 1} if architecture == "transformer" else {}
    model = build_model(architecture, **overrides)
    before = model.forward(sequences, lengths, months, training=False)

    path = model.save_checkpoint(tmp_path / "model.npz",
                                  {"species": "zebra", "max_length": 10,
                                   "architecture": architecture})
    restored, meta = load_checkpoint(path)
    after = restored.forward(sequences, lengths, months, training=False)

    assert meta["model_type"] == type(model).__name__
    assert np.allclose(before[0].data, after[0].data)
    assert np.allclose(before[1].data, after[1].data)


def test_checkpoint_rejects_a_shape_mismatch(tmp_path, detections):
    model = LSTMBehaviourModel(hidden_size=8)
    path = model.save_checkpoint(tmp_path / "small.npz", {"species": "zebra"})
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key != "__meta__"}
    with pytest.raises(ValueError, match="Shape mismatch"):
        LSTMBehaviourModel(hidden_size=16).load_state(arrays)


def test_training_reduces_the_loss(detections):
    examples = build_training_set(detections, max_length=10, species="zebra")
    sequences, lengths, activity, movement, months = examples_to_tensors(examples)
    model = LSTMBehaviourModel(hidden_size=8, num_layers=1)
    train_model(model, sequences, lengths, activity, movement, months,
                num_epochs=30, verbose=False)
    history = model.training_history
    assert history[-1]["loss"] < history[0]["loss"]


def test_evaluation_reports_per_class_support(detections):
    examples = build_training_set(detections, max_length=10, species="zebra")
    sequences, lengths, activity, movement, months = examples_to_tensors(examples)
    model = LSTMBehaviourModel(hidden_size=8, num_layers=1)
    metrics = evaluate_model(model, sequences, lengths, activity, movement,
                              months, verbose=False)

    assert 0.0 <= metrics["movement_accuracy"] <= 100.0
    assert set(metrics["movement_per_class"]) == set(labelling.MOVEMENT_CLASSES)
    total_support = sum(entry["support"]
                        for entry in metrics["movement_per_class"].values())
    assert total_support == len(examples)


def test_confusion_matrix_and_recall_agree():
    matrix = confusion_matrix([0, 0, 1, 2], [0, 1, 1, 2], 3)
    assert matrix.sum() == 4
    report = per_class_recall(matrix, ["a", "b", "c"])
    assert report["a"]["recall"] == pytest.approx(0.5)
    assert report["b"]["support"] == 1


def test_service_predicts_one_pattern_per_camera(tmp_path, detections, monkeypatch):
    from wildlife_monitor.pipeline2 import inference

    examples = build_training_set(detections, max_length=10, species="zebra")
    sequences, lengths, activity, movement, months = examples_to_tensors(examples)
    model = LSTMBehaviourModel(hidden_size=8, num_layers=1)
    train_model(model, sequences, lengths, activity, movement, months,
                num_epochs=5, verbose=False)
    model.save_checkpoint(tmp_path / "zebra_lstm.npz",
                           {"species": "zebra", "architecture": "lstm",
                            "max_length": 10})

    monkeypatch.setattr(inference, "BEHAVIOUR_DIR", tmp_path)
    service = BehaviourService.load("zebra", "lstm")
    patterns = service.predict(detections, "zebra")

    assert len(patterns) == 3
    assert {p.camera_id for p in patterns} == {"HIGH", "PEAK", "THIN"}
    for pattern in patterns:
        assert pattern.activity_class in labelling.ACTIVITY_CLASSES
        assert pattern.movement_class in labelling.MOVEMENT_CLASSES
        assert pattern.social_class in labelling.SOCIAL_CLASSES
        assert 0.0 <= pattern.movement_confidence <= 1.0
        assert 1 <= pattern.peak_month <= 12

    high = next(p for p in patterns if p.camera_id == "HIGH")
    assert high.sequence_truncated and high.detection_count == 120
