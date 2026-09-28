"""
Training and evaluation for the behavioural models (Package P3).

This module owns the training loop and the evaluation metrics only. The label
rules live in :mod:`wildlife_monitor.pipeline2.labelling` and sequence
construction in :mod:`wildlife_monitor.pipeline2.sequence_builder`, so that
inference can reuse both without importing an optimiser.
"""

from __future__ import annotations

import gc
import random

import numpy as np
import pandas as pd

from wildlife_monitor.pipeline2.autodiff import (
    Adam, softmax_cross_entropy, clip_grad_norm,
)
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, ACTIVITY_TO_IDX, MOVEMENT_TO_IDX,
    classify_activity, classify_movement,
)
from wildlife_monitor.pipeline2.sequence_builder import SequenceBuilder

# Re-exported for callers that historically imported them from this module.
__all__ = [
    "ACTIVITY_CLASSES", "MOVEMENT_CLASSES", "ACTIVITY_TO_IDX", "MOVEMENT_TO_IDX",
    "build_training_set", "build_cross_species_set", "predict_in_chunks",
    "split_train_test",
    "split_by_species", "leave_one_species_out", "examples_to_tensors",
    "train_model", "evaluate_model", "confusion_matrix", "per_class_recall",
]


def build_training_set(source, max_length: int = 40,
                        species: str = "") -> list[dict]:
    """Build one labelled training example per camera.

    ``source`` may be a detections frame (read from the database) or a path to
    a detections CSV, so the same function serves the live system and any
    archived CSV a reviewer wants to reproduce results from.
    """
    frame = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    builder = SequenceBuilder(max_length)
    stats = builder.camera_statistics(frame)
    grouped = builder.group_by_site(frame)
    sequences = builder.build_all(frame, species)

    examples = []
    for sequence in sequences:
        camera_df = grouped[sequence.camera_id]
        activity_label = classify_activity(camera_df)
        movement_label = classify_movement(
            sequence.camera_id, stats["fidelity"],
            stats["temporal_concentration"], stats["detection_counts"])
        examples.append({
            "camera_id": sequence.camera_id,
            "species": species,
            "vectors": sequence.vectors,
            "real_length": sequence.real_length,
            "detection_count": sequence.detection_count,
            "social_label": sequence.social_label,
            "activity_label": activity_label,
            "movement_label": movement_label,
            "activity_idx": ACTIVITY_TO_IDX[activity_label],
            "movement_idx": MOVEMENT_TO_IDX[movement_label],
            "month_features": sequence.month_features,
        })
    return examples


def build_cross_species_set(sources: dict, max_length: int = 40) -> list[dict]:
    """Pool labelled examples from several species into one training set.

    ``sources`` maps a species name to its detections frame or CSV path.

    Labels are computed **per species, before pooling**. This matters. Both
    movement statistics are compared against ``quantile(0.75)`` of the other
    cameras, so pooling the frames first would compute that quantile across a
    mixture of animals and silently relabel every camera. Computing per species
    and then pooling keeps every label identical to the per-species runs, which
    is what makes the two sets of results comparable.
    """
    examples: list[dict] = []
    for species in sorted(sources):
        for example in build_training_set(sources[species], max_length,
                                           species=species):
            example.setdefault("species", species)
            examples.append(example)
    return examples


def species_in(examples) -> list[str]:
    """Every species present in a pooled example set, in a stable order."""
    return sorted({str(example.get("species", "")) for example in examples}
                  - {""})


def split_train_test(examples, test_fraction=0.2, seed=42):
    """Split by camera. Test cameras are unseen sites of a *known* species."""
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    num_test = max(1, round(len(shuffled) * test_fraction))
    return shuffled[num_test:], shuffled[:num_test]


def split_by_species(examples, held_out) -> tuple[list[dict], list[dict]]:
    """Split by species. Test examples are of a species never trained on.

    This is a strictly harder test than :func:`split_train_test`. There, the
    model has seen the same animal at other sites. Here it has never seen the
    animal at all, so anything it gets right came from the shape of the
    detection history rather than from memorising one species.
    """
    if isinstance(held_out, str):
        held_out = [held_out]
    excluded = {str(name).strip().lower() for name in held_out}

    train, test = [], []
    for example in examples:
        species = str(example.get("species", "")).strip().lower()
        (test if species in excluded else train).append(example)
    return train, test


def leave_one_species_out(examples):
    """Yield ``(held_out_species, train, test)`` once per species.

    With only a handful of species a single held-out split rests on whichever
    species happens to be chosen. Rotating through all of them gives one
    generalisation result per species instead of one overall, which is both
    more informative and harder to get lucky on.
    """
    for species in species_in(examples):
        train, test = split_by_species(examples, species)
        if train and test:
            yield species, train, test


def examples_to_tensors(examples):
    sequences = np.array([ex["vectors"] for ex in examples], dtype=np.float64)
    lengths = np.array([ex["real_length"] for ex in examples], dtype=np.int64)
    activity_idxs = np.array([ex["activity_idx"] for ex in examples], dtype=np.int64)
    movement_idxs = np.array([ex["movement_idx"] for ex in examples], dtype=np.int64)
    month_features = np.array([ex["month_features"] for ex in examples], dtype=np.float64)
    return sequences, lengths, activity_idxs, movement_idxs, month_features


def _step(model, optimizer, sequences, lengths, activity, movement, months,
          max_grad_norm: float):
    """One forward, backward and parameter update. Returns the three losses."""
    optimizer.zero_grad()
    activity_logits, movement_logits = model.forward(
        sequences, lengths, months, training=True)
    activity_loss = softmax_cross_entropy(activity_logits, activity)
    movement_loss = softmax_cross_entropy(movement_logits, movement)
    total_loss = activity_loss + movement_loss
    total_loss.backward()
    clip_grad_norm(model.parameters(), max_grad_norm)
    optimizer.step()
    return (float(total_loss.data), float(activity_loss.data),
            float(movement_loss.data))


def train_model(model, train_sequences, train_lengths, train_activity, train_movement,
                 train_month_features, num_epochs=100, learning_rate=0.001,
                 max_grad_norm=5.0, verbose=True, batch_size=None, seed=42):
    """Train for ``num_epochs``, full-batch by default.

    ``batch_size`` splits each epoch into mini-batches instead. This matters
    for memory rather than for optimisation. The autodiff graph holds every
    intermediate tensor until the backward pass completes, and its size grows
    with the number of sequences processed together. Pooling five species
    raises that count from roughly a hundred cameras to several hundred, which
    is enough to exhaust memory on a laptop.

    The default stays ``None`` so single-species runs behave exactly as before
    and their published results remain reproducible.
    """
    optimizer = Adam(model.parameters(), lr=learning_rate)
    history = []
    total_examples = len(train_sequences)
    batching = bool(batch_size) and batch_size < total_examples
    generator = np.random.default_rng(seed)

    for epoch in range(num_epochs):
        if not batching:
            loss, activity_loss, movement_loss = _step(
                model, optimizer, train_sequences, train_lengths,
                train_activity, train_movement, train_month_features,
                max_grad_norm)
        else:
            order = generator.permutation(total_examples)
            loss = activity_loss = movement_loss = 0.0
            for start in range(0, total_examples, batch_size):
                rows = order[start:start + batch_size]
                share = len(rows) / total_examples
                batch_loss, batch_activity, batch_movement = _step(
                    model, optimizer, train_sequences[rows],
                    train_lengths[rows], train_activity[rows],
                    train_movement[rows], train_month_features[rows],
                    max_grad_norm)
                loss += batch_loss * share
                activity_loss += batch_activity * share
                movement_loss += batch_movement * share

        history.append({
            "epoch": epoch + 1,
            "loss": loss,
            "activity_loss": activity_loss,
            "movement_loss": movement_loss,
        })
        if (epoch + 1) % 10 == 0:
            if verbose:
                print(f"  Epoch {epoch + 1:>3}/{num_epochs}  "
                      f"loss={loss:.4f}  "
                      f"(activity={activity_loss:.4f}, "
                      f"movement={movement_loss:.4f})")
            gc.collect()

    model.training_history = history
    return model


def confusion_matrix(true_idxs, pred_idxs, num_classes: int) -> np.ndarray:
    """Rows are true classes, columns predicted."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(true_idxs, pred_idxs):
        matrix[int(true_idx), int(pred_idx)] += 1
    return matrix


def per_class_recall(matrix: np.ndarray, classes: list[str]) -> dict[str, dict]:
    """Recall and support per class, so a rare class cannot hide in the mean."""
    report = {}
    for index, name in enumerate(classes):
        support = int(matrix[index].sum())
        correct = int(matrix[index, index])
        report[name] = {
            "support": support,
            "correct": correct,
            "recall": round(correct / support, 4) if support else None,
        }
    return report


def predict_in_chunks(model, sequences, lengths, months,
                      chunk_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass in chunks, returning raw logits.

    A forward pass builds the same autodiff graph whether or not a backward
    pass follows, so evaluating a whole held-out species at once costs as much
    memory as training on it. Chunking keeps the peak bounded.
    """
    activity_parts, movement_parts = [], []
    for start in range(0, len(sequences), chunk_size):
        stop = start + chunk_size
        activity_logits, movement_logits = model.forward(
            sequences[start:stop], lengths[start:stop], months[start:stop],
            training=False)
        activity_parts.append(np.asarray(activity_logits.data))
        movement_parts.append(np.asarray(movement_logits.data))
        gc.collect()
    return np.concatenate(activity_parts), np.concatenate(movement_parts)


def evaluate_model(model, test_sequences, test_lengths, test_activity, test_movement,
                    test_month_features, verbose=True, chunk_size=64) -> dict:
    """Evaluate on held-out cameras and return a full metrics dictionary."""
    activity_data, movement_data = predict_in_chunks(
        model, test_sequences, test_lengths, test_month_features, chunk_size)
    activity_preds = np.argmax(activity_data, axis=1)
    movement_preds = np.argmax(movement_data, axis=1)

    if verbose:
        print("Activity predictions vs true labels:")
        for true_idx, pred_idx in zip(test_activity, activity_preds):
            status = "[correct]" if true_idx == pred_idx else "[WRONG]"
            print(f"  true={ACTIVITY_CLASSES[true_idx]:<12} "
                  f"predicted={ACTIVITY_CLASSES[pred_idx]:<12} {status}")

    activity_accuracy = float((activity_preds == test_activity).mean() * 100)
    if verbose:
        print(f"\nOverall activity accuracy: {activity_accuracy:.1f}%\n")
        print("Movement predictions vs true labels:")
        for true_idx, pred_idx in zip(test_movement, movement_preds):
            status = "[correct]" if true_idx == pred_idx else "[WRONG]"
            print(f"  true={MOVEMENT_CLASSES[true_idx]:<12} "
                  f"predicted={MOVEMENT_CLASSES[pred_idx]:<12} {status}")

    movement_accuracy = float((movement_preds == test_movement).mean() * 100)
    if verbose:
        print(f"\nOverall movement accuracy: {movement_accuracy:.1f}%")

    activity_matrix = confusion_matrix(test_activity, activity_preds,
                                        len(ACTIVITY_CLASSES))
    movement_matrix = confusion_matrix(test_movement, movement_preds,
                                        len(MOVEMENT_CLASSES))
    return {
        "test_cameras": int(len(test_activity)),
        "activity_accuracy": round(activity_accuracy, 2),
        "movement_accuracy": round(movement_accuracy, 2),
        "activity_per_class": per_class_recall(activity_matrix, ACTIVITY_CLASSES),
        "movement_per_class": per_class_recall(movement_matrix, MOVEMENT_CLASSES),
        "activity_confusion": activity_matrix.tolist(),
        "movement_confusion": movement_matrix.tolist(),
    }
