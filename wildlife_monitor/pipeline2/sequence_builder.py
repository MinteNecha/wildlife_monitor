"""
Set sequences ready for the LSTM/Transformer models to learn.
Group detection records int per camera sequence
"""
from __future__ import annotations
from wildlife_monitor.pipeline2.feature_extractor import build_feature_vector
import pandas as pd

def group_by_camera(detections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Split detections into camera site groups, sorted by time
    """
    sequences: dict[str, pd.DataFrame] = {}

    for camera_id, camera_rows in detections.groupby("camera_id"):
        sorted_rows = camera_rows.sort_values("timestamp")
        sequences[camera_id] = sorted_rows.reset_index(drop=True) #dropping original index because of the new sorting

    return sequences

def classify_social_structure(camera_sequence: pd.DataFrame) -> str:
    """
    Using the max count, compute a social structure label for that value
    """
    max_count = camera_sequence["instance_count"].max()

    if max_count <= 1:
        return "solitary"
    elif max_count <= 5:
        return "small group"
    else:
        return "large herd"

def build_sequence(camera_sequence: pd.DataFrame, max_length: int,) -> tuple[list[list[float]], int, str]:
    social_label = classify_social_structure(camera_sequence)

    vectors = [
        build_feature_vector(timestamp = row["timestamp"], instance_count = row["instance_count"],)
        for _, row in camera_sequence.iterrows()
    ]

    real_length = len(vectors)

    if real_length >= max_length:
        padded_vectors = vectors[:max_length]
        real_length = max_length
    else:
        num_features = len(vectors[0]) if vectors else 5
        padding_needed = max_length - real_length
        padded_vectors = vectors + [[0.0] * num_features] * padding_needed

    return padded_vectors, real_length, social_label