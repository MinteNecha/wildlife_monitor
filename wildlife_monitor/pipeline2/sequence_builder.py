"""
Detection sequences for the behavioural models (Package P3).

A *sequence* is one camera's detection history for one species, ordered by
time. It is the unit the behavioural models consume: one sequence in, one set
of behavioural classifications out.

Two things matter about the ``max_length`` cap applied here:

* It governs **model input only**. Every label and every summary statistic is
  computed from the camera's complete, uncapped history.
* It is only reached by a minority of cameras — overwhelmingly the
  high-traffic, territorial ones. Measured across the five species in this
  project, 11-21% of cameras exceed 40 detections, and 94-100% of those are
  territorial.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from wildlife_monitor.pipeline2.feature_extractor import (
    TemporalFeatureExtractor, NUM_FEATURES as EXTENDED_NUM_FEATURES,
    LEGACY_NUM_FEATURES,
)
from wildlife_monitor.pipeline2.labelling import (
    classify_social_structure, compute_site_fidelity,
    compute_temporal_concentration, compute_monthly_distribution,
)

NUM_FEATURES = EXTENDED_NUM_FEATURES
NUM_MONTH_FEATURES = 12


@dataclass
class DetectionSequence:
    """One camera's ordered detections for one species, ready for a model."""

    camera_id: str
    species: str
    vectors: list[list[float]]        # padded / truncated to max_length
    real_length: int                  # usable timesteps in ``vectors``
    detection_count: int              # complete history, ignoring the cap
    social_label: str                 # rule-derived, from max instance count
    month_features: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_MONTH_FEATURES))
    start_time: str = ""
    end_time: str = ""

    @property
    def truncated(self) -> bool:
        """True when the cap discarded part of this camera's history."""
        return self.detection_count > self.real_length


class SequenceBuilder:
    """Builds :class:`DetectionSequence` objects from detection records.

    The per-camera summary statistics (site fidelity, temporal concentration,
    monthly distribution) are computed once over the whole frame and reused for
    every sequence, since each depends on the full population of cameras.
    """

    def __init__(self, max_length: int = 40, legacy_features: bool = False) -> None:
        self.max_length = max_length
        self.extractor = TemporalFeatureExtractor(legacy=legacy_features)

    @property
    def num_features(self) -> int:
        return self.extractor.num_features

    @classmethod
    def for_model(cls, max_length: int, num_features: int) -> "SequenceBuilder":
        """Builder matching a trained model's input width."""
        return cls(max_length, legacy_features=(num_features == LEGACY_NUM_FEATURES))

    def group_by_site(self, detections: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Split detections into per-camera frames, each sorted by time."""
        groups: dict[str, pd.DataFrame] = {}
        for camera_id, rows in detections.groupby("camera_id"):
            groups[str(camera_id)] = rows.sort_values("timestamp").reset_index(drop=True)
        return groups

    def build(self, camera_sequence: pd.DataFrame, camera_id: str,
              species: str = "", month_features: np.ndarray | None = None
              ) -> DetectionSequence:
        """Build one sequence from a single camera's detections."""
        vectors = self.extractor.extract(camera_sequence)
        detection_count = len(vectors)

        if detection_count >= self.max_length:
            padded = vectors[:self.max_length]
            real_length = self.max_length
        else:
            padding = [[0.0] * self.num_features] * (self.max_length - detection_count)
            padded = vectors + padding
            real_length = detection_count

        timestamps = camera_sequence["timestamp"].astype(str)
        return DetectionSequence(
            camera_id=camera_id,
            species=species or str(camera_sequence.get("species", pd.Series([""])).iloc[0]),
            vectors=padded,
            real_length=real_length,
            detection_count=detection_count,
            social_label=classify_social_structure(camera_sequence),
            month_features=(np.zeros(NUM_MONTH_FEATURES)
                            if month_features is None else month_features),
            start_time=timestamps.iloc[0] if detection_count else "",
            end_time=timestamps.iloc[-1] if detection_count else "",
        )

    def build_all(self, detections: pd.DataFrame,
                  species: str = "") -> list[DetectionSequence]:
        """Build one sequence per camera present in ``detections``."""
        monthly = compute_monthly_distribution(detections)
        return [
            self.build(camera_df, camera_id, species,
                       monthly.get(camera_id, np.zeros(NUM_MONTH_FEATURES)))
            for camera_id, camera_df in self.group_by_site(detections).items()
        ]

    @staticmethod
    def camera_statistics(detections: pd.DataFrame) -> dict[str, pd.Series]:
        """Per-camera statistics the movement rule needs, computed once."""
        return {
            "fidelity": compute_site_fidelity(detections),
            "temporal_concentration": compute_temporal_concentration(detections),
            "detection_counts": detections.groupby("camera_id").size(),
        }


def sequences_to_arrays(sequences: list[DetectionSequence]
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack sequences into the arrays a model's ``forward`` expects."""
    vectors = np.array([s.vectors for s in sequences], dtype=np.float64)
    lengths = np.array([s.real_length for s in sequences], dtype=np.int64)
    months = np.array([s.month_features for s in sequences], dtype=np.float64)
    return vectors, lengths, months


# ── Backwards-compatible function API ─────────────────────────────────────────
# Kept so existing scripts and notebooks continue to work. New code should use
# SequenceBuilder directly.

def group_by_camera(detections: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split detections into camera site groups, sorted by time."""
    return SequenceBuilder().group_by_site(detections)


def build_sequence(camera_sequence: pd.DataFrame, max_length: int,
                   legacy_features: bool = False
                   ) -> tuple[list[list[float]], int, str]:
    """Legacy shim returning ``(padded_vectors, real_length, social_label)``."""
    sequence = SequenceBuilder(max_length, legacy_features).build(
        camera_sequence, camera_id="")
    return sequence.vectors, sequence.real_length, sequence.social_label
