"""
Temporal feature extraction for the behavioural models (Package P3, FR4).

Turns one detection into a fixed-length vector of numbers. The proposal
specifies hour of day, day of year, time since the previous detection, camera
identity and habitat type. Four of those five are implemented here. Camera
identity is deliberately excluded, and that decision is worth stating plainly:

    Every model in this project is evaluated on *held-out cameras* — sites the
    model never saw during training. A raw camera identifier cannot generalise
    to a camera that was not in the training set, so including it would add a
    feature that is either useless at evaluation time or, worse, lets the model
    memorise training sites and look better than it is. Camera *context* that
    does generalise — habitat type — is included instead.

Cyclical values (hour, day of year) are encoded as sine/cosine pairs so that
23:00 and 00:00 sit next to each other rather than at opposite ends of a line.
"""

from __future__ import annotations

import math
from datetime import datetime

# Habitat vocabulary, matching the published Serengeti camera grid. Fixed and
# small, so a one-hot encoding stays cheap and stable across runs.
HABITAT_CLASSES = ["open_grassland", "woodland", "riverine", "kopje", "unknown"]
HABITAT_TO_IDX = {name: i for i, name in enumerate(HABITAT_CLASSES)}

TIMESTAMP_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M")

# Feature layout, in order. NUM_FEATURES is derived from it so a model's input
# width and this list can never drift apart.
FEATURE_NAMES = [
    "sin_hour", "cos_hour", "sin_day", "cos_day",
    "log_count", "log_interval_hours",
    *[f"habitat_{name}" for name in HABITAT_CLASSES],
]
NUM_FEATURES = len(FEATURE_NAMES)

# The original five-feature layout, kept so checkpoints trained before habitat
# and interval features were added can still be rebuilt and served.
LEGACY_FEATURE_NAMES = ["sin_hour", "cos_hour", "sin_day", "cos_day", "log_count"]
LEGACY_NUM_FEATURES = len(LEGACY_FEATURE_NAMES)


def parse_timestamp(timestamp) -> datetime | None:
    """Parse a timestamp string, returning None when it cannot be read."""
    text = str(timestamp).strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except (ValueError, TypeError):
            continue
    return None


def extract_hour(timestamp) -> float:
    """Hour of day as a float. Defaults to midday when unparseable."""
    parsed = parse_timestamp(timestamp)
    return 12.0 if parsed is None else parsed.hour + parsed.minute / 60.0


def extract_day_of_year(timestamp) -> float:
    """Day of year. Defaults to roughly mid-year when unparseable."""
    parsed = parse_timestamp(timestamp)
    return 182.0 if parsed is None else float(parsed.timetuple().tm_yday)


def encode_cyclical(value: float, period: float) -> tuple[float, float]:
    """Place a cyclical value on a circle so its ends meet."""
    radians = 2 * math.pi * value / period
    return math.sin(radians), math.cos(radians)


def cyclic_hour_encoding(hour: float) -> tuple[float, float]:
    return encode_cyclical(hour, 24.0)


def cyclcic_day_encoding(day_of_year: float) -> tuple[float, float]:
    """Retained under its original (misspelled) name for compatibility."""
    return encode_cyclical(day_of_year, 365.0)


cyclic_day_encoding = cyclcic_day_encoding


def compute_intervals(timestamps: list) -> list[float]:
    """Hours since the previous detection, 0.0 for the first in a sequence."""
    intervals = [0.0]
    parsed = [parse_timestamp(value) for value in timestamps]
    for previous, current in zip(parsed, parsed[1:]):
        if previous is None or current is None:
            intervals.append(0.0)
        else:
            gap = (current - previous).total_seconds() / 3600.0
            intervals.append(max(gap, 0.0))
    return intervals


def encode_habitat(habitat: str) -> list[float]:
    """One-hot habitat, falling back to 'unknown' for unseen values."""
    encoding = [0.0] * len(HABITAT_CLASSES)
    index = HABITAT_TO_IDX.get(str(habitat).strip().lower(),
                                HABITAT_TO_IDX["unknown"])
    encoding[index] = 1.0
    return encoding


class TemporalFeatureExtractor:
    """Builds per-detection feature vectors for a camera sequence (FR4).

    Set ``legacy=True`` to emit the original five-feature vector, which is what
    checkpoints trained before the interval and habitat features expect.
    """

    def __init__(self, legacy: bool = False) -> None:
        self.legacy = legacy

    @property
    def feature_names(self) -> list[str]:
        return LEGACY_FEATURE_NAMES if self.legacy else FEATURE_NAMES

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    def extract_one(self, timestamp, instance_count: int,
                    interval_hours: float = 0.0,
                    habitat: str = "unknown") -> list[float]:
        """One detection -> one feature vector."""
        sin_hour, cos_hour = cyclic_hour_encoding(extract_hour(timestamp))
        sin_day, cos_day = cyclic_day_encoding(extract_day_of_year(timestamp))
        try:
            count = float(instance_count)
        except (TypeError, ValueError):
            count = 0.0
        log_count = math.log(max(count, 0.0) + 1)

        if self.legacy:
            return [sin_hour, cos_hour, sin_day, cos_day, log_count]

        return [sin_hour, cos_hour, sin_day, cos_day, log_count,
                math.log(max(interval_hours, 0.0) + 1), *encode_habitat(habitat)]

    def extract(self, camera_sequence) -> list[list[float]]:
        """A camera's time-ordered detections -> a list of feature vectors.

        Intervals are computed across the whole sequence here rather than per
        row, because the gap between detections is a property of the pair, not
        of either detection alone.
        """
        timestamps = list(camera_sequence["timestamp"])
        intervals = compute_intervals(timestamps)
        counts = list(camera_sequence["instance_count"])
        if "habitat_type" in camera_sequence:
            habitats = list(camera_sequence["habitat_type"])
        else:
            habitats = ["unknown"] * len(timestamps)

        return [
            self.extract_one(timestamp, count, interval, habitat)
            for timestamp, count, interval, habitat
            in zip(timestamps, counts, intervals, habitats)
        ]


def build_feature_vector(timestamp: str, instance_count: int,
                          interval_hours: float = 0.0,
                          habitat: str = "unknown",
                          legacy: bool = False) -> list[float]:
    """Functional wrapper around :class:`TemporalFeatureExtractor`."""
    return TemporalFeatureExtractor(legacy).extract_one(
        timestamp, instance_count, interval_hours, habitat)
