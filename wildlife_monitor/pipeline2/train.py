"""
Training loop for Pipeline 2 behaviour models
"""
from __future__ import annotations
import torch 
import pandas as pd
import random

from wildlife_monitor.pipeline2.sequence_builder import group_by_camera, build_sequence
from wildlife_monitor.pipeline2.feature_extractor import extract_hour

ACTIVITY_CLASSES = ["diurnal", "nocturnal", "crepuscular"]
MOVEMENT_CLASSES = ["migratory", "territorial", "nomadic"]

def classify_activity(camera_sequence: pd.DataFrame) -> str:
    hours = [extract_hour(ts) for ts in camera_sequence["timestamp"]]

    categories = []
    for h in hours:
        if 6 <= h < 18:
            categories.append("diurnal")
        elif h < 5 or h >= 19:
            categories.append("nocturnal")
        else:
            categories.append("crepuscular")

    return max(set(categories), key=categories.count)

ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
MOVEMENT_TO_IDX = {name: i for i, name in enumerate(MOVEMENT_CLASSES)}


def classify_movement(camera_sequence: pd.DataFrame, threshold_days: float = 30.0) -> str:
    """
    Provisional movement strategy heuristic, based only on how spread
    out in time one camera's detections are. See project notes: this
    is the weakest of the three labels and a known limitation, since
    genuine movement strategy requires comparing patterns ACROSS
    camera sites, not just within one. "migratory" is intentionally
    not produced by this rule; only "territorial" or "nomadic".
    """
    timestamps = pd.to_datetime(camera_sequence["timestamp"], errors="coerce")
    timestamps = timestamps.dropna()

    if len(timestamps) < 2:
        return "territorial"

    span_days = (timestamps.max() - timestamps.min()).total_seconds() / 86400.0

    return "territorial" if span_days <= threshold_days else "nomadic"


def build_training_set(
    detections_csv: str,
    max_length: int = 40,
) -> list[dict]:
    """
    Read a Pipeline 1 detections CSV and build one training example
    per camera site, each with its padded sequence, real length, and
    the three labels (social structure, activity, movement).
    """
    df = pd.read_csv(detections_csv)
    sequences = group_by_camera(df)

    examples = []
    for camera_id, camera_df in sequences.items():
        padded_vectors, real_length, social_label = build_sequence(
            camera_df, max_length
        )
        activity_label = classify_activity(camera_df)
        movement_label = classify_movement(camera_df)

        examples.append({
            "camera_id": camera_id,
            "vectors": padded_vectors,
            "real_length": real_length,
            "social_label": social_label,
            "activity_label": activity_label,
            "activity_idx": ACTIVITY_TO_IDX[activity_label],
            "movement_label": movement_label,
            "movement_idx": MOVEMENT_TO_IDX[movement_label],
        })

    return examples

def split_train_test(examples: list[dict], test_fraction: float=0.2, seed: int=42,) -> tuple[list[dict], list[dict]]:
    """
    Randomly split camera examples into a training group and a testing group.
    """
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)

    num_test = max(1, round(len(shuffled)*test_fraction))
    test_examples = shuffled[:num_test]
    train_examples = shuffled[num_test:]

    return train_examples, test_examples