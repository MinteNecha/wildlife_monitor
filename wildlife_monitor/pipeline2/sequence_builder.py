"""
Set sequences ready for the LSTM/Transformer models to learn.
Group detection records int per camera sequence
"""

from __future__ import annotations
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