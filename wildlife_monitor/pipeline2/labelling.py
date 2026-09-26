"""
Behavioural label taxonomy and the rules that derive labels from detections
(Package P3).

These rules turn raw detection records into the three behavioural categories
the proposal defines. They live apart from ``train.py`` because both training
and inference need them: training needs them to produce targets, and the
dashboard needs them to describe what a prediction means and to derive the
social-structure label.

Every rule reads a camera's *complete* detection history. None of them are
affected by the ``max_length`` cap applied when a sequence is fed to a model —
that cap governs model input only, never the labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ACTIVITY_CLASSES = ["diurnal", "nocturnal", "crepuscular"]
MOVEMENT_CLASSES = ["migratory", "territorial", "nomadic"]
SOCIAL_CLASSES = ["solitary", "small group", "large herd"]

ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
MOVEMENT_TO_IDX = {name: i for i, name in enumerate(MOVEMENT_CLASSES)}
SOCIAL_TO_IDX = {name: i for i, name in enumerate(SOCIAL_CLASSES)}

# Plain-English descriptions, shown next to predictions in the dashboard so a
# non-technical reader knows what each class actually asserts.
CLASS_DESCRIPTIONS = {
    "diurnal": "Active mainly during daylight hours (06:00-18:00).",
    "nocturnal": "Active mainly after dark (19:00-05:00).",
    "crepuscular": "Active mainly around dawn and dusk.",
    "migratory": "Presence concentrated into a narrow seasonal window, "
                 "consistent with passage rather than residency.",
    "territorial": "High site fidelity — this camera accounts for a large "
                   "share of all detections for the species.",
    "nomadic": "Presence spread thinly across the season with no strong "
               "site or seasonal concentration.",
    "solitary": "Never more than one individual seen in a single frame.",
    "small group": "Up to five individuals seen in a single frame.",
    "large herd": "More than five individuals seen in a single frame.",
}


def extract_hour(timestamp) -> float:
    """Hour-of-day for one timestamp, delegating to the feature extractor."""
    from wildlife_monitor.pipeline2.feature_extractor import (
        extract_hour as _extract_hour,
    )
    return _extract_hour(timestamp)


def classify_activity(camera_sequence: pd.DataFrame) -> str:
    """Majority activity window across a camera's detections."""
    hours = [extract_hour(ts) for ts in camera_sequence["timestamp"]]
    categories = []
    for hour in hours:
        if 6 <= hour < 18:
            categories.append("diurnal")
        elif hour < 5 or hour >= 19:
            categories.append("nocturnal")
        else:
            categories.append("crepuscular")
    return max(set(categories), key=categories.count)


def classify_social_structure(camera_sequence: pd.DataFrame) -> str:
    """Group size band, taken from the largest single-frame instance count."""
    max_count = camera_sequence["instance_count"].max()
    if max_count <= 1:
        return "solitary"
    if max_count <= 5:
        return "small group"
    return "large herd"


def compute_site_fidelity(all_detections: pd.DataFrame) -> pd.Series:
    """
    For each camera, what fraction of ALL detections (across every camera)
    happened at that one camera. High fraction = animal favours that site.
    """
    total_detections = len(all_detections)
    camera_counts = all_detections.groupby("camera_id").size()
    return camera_counts / total_detections


def compute_temporal_concentration(all_detections: pd.DataFrame) -> pd.Series:
    """
    For each camera, the largest single-month share of that camera's own
    detections. High share = presence bunched into a narrow window
    (migratory-style passage) rather than spread across the year.
    """
    months = pd.to_datetime(all_detections["timestamp"], errors="coerce").dt.month
    by_month = all_detections.assign(_month=months).groupby(["camera_id", "_month"]).size()
    totals = all_detections.groupby("camera_id").size()
    busiest_month = by_month.groupby("camera_id").max()
    return (busiest_month / totals).fillna(0.0)


def compute_monthly_distribution(all_detections: pd.DataFrame,
                                  num_months: int = 12) -> dict:
    """
    For each camera, the share of ITS OWN detections falling in each calendar
    month (Jan..Dec), computed from that camera's full, uncapped detection
    history. Gives a model the shape of a camera's seasonal presence, where
    :func:`classify_movement` keeps only the single busiest-month share.
    """
    months = pd.to_datetime(all_detections["timestamp"], errors="coerce").dt.month
    working = all_detections.assign(_month=months)
    counts = working.groupby(["camera_id", "_month"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=range(1, num_months + 1), fill_value=0)
    totals = counts.sum(axis=1)
    shares = counts.div(totals.replace(0, 1), axis=0)
    return {camera_id: shares.loc[camera_id].to_numpy(dtype=np.float64)
            for camera_id in shares.index}


def classify_movement(camera_id: str, fidelity: pd.Series,
                       temporal_concentration: pd.Series,
                       detection_counts: pd.Series,
                       territorial_percentile: float = 0.75,
                       migratory_percentile: float = 0.75,
                       min_detections_for_migratory: int = 3) -> str:
    """
    Movement strategy for one camera.

    Migratory is tested first: a camera whose detections are unusually
    concentrated in one month reads as passage. Everything else splits on site
    fidelity, with the most-favoured sites called territorial and the rest
    nomadic.
    """
    if detection_counts.get(camera_id, 0) >= min_detections_for_migratory:
        migratory_threshold = temporal_concentration.quantile(migratory_percentile)
        if temporal_concentration.get(camera_id, 0.0) >= migratory_threshold:
            return "migratory"

    threshold_value = fidelity.quantile(territorial_percentile)
    return "territorial" if fidelity.get(camera_id, 0.0) >= threshold_value else "nomadic"
