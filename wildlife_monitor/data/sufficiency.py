"""
Data sufficiency assessment (Package P1).

Answers one question before any behavioural result is shown: *can this data
actually support the thing we are about to claim?*

This exists because the behavioural rules degrade silently rather than
loudly. ``classify_movement`` compares each camera against the 75th percentile
of the other cameras, so it always produces a confident-looking answer — even
when that answer is meaningless. A one-month deployment gives every camera
100% of its detections in one month, the percentile threshold lands on top of
all of them, and every single camera is labelled migratory. Nothing errors.
The dashboard would present it as fact.

So each behavioural output is checked against the data it actually needs:

    activity timing   hours of day, so it needs enough detections per camera
                      but no particular time span
    movement strategy seasonal contrast, so it needs months of coverage AND
                      enough cameras for a percentile to mean anything
    social structure  group sizes, so it only needs instance counts

The thresholds below are judgements, not discoveries, and are documented as
such. They are set from where each mechanism visibly breaks rather than from
a statistical power calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Status = Literal["supported", "limited", "unsupported"]

# Where each mechanism breaks. Stated as (limited_below, supported_at).
MIN_SPAN_MONTHS = (3, 6)        # seasonal contrast needs months of coverage
MIN_CAMERAS = (10, 30)          # percentile thresholds need a population
MIN_DETECTIONS_PER_CAMERA = (3, 10)   # 3 is the coded migratory floor
MIN_DETECTIONS_FOR_ACTIVITY = (5, 10)

STATUS_LABELS = {
    "supported": "Supported",
    "limited": "Limited — treat with caution",
    "unsupported": "Not supported",
}


def _months(value: float) -> str:
    """Readable coverage: days under a month, months above it.

    A nine-day deployment rounding to "0 months" reads as a bug rather than as
    the (accurate) statement that there is almost no coverage.
    """
    if value < 1:
        days = max(int(round(value * 30.44)), 0)
        return f"{days} day{'' if days == 1 else 's'}"
    return f"{value:.0f} month{'' if round(value) == 1 else 's'}"


def _grade(value: float, thresholds: tuple[float, float]) -> Status:
    limited_below, supported_at = thresholds
    if value < limited_below:
        return "unsupported"
    if value < supported_at:
        return "limited"
    return "supported"


def _worst(*statuses: Status) -> Status:
    for level in ("unsupported", "limited", "supported"):
        if level in statuses:
            return level
    return "supported"


@dataclass
class Capability:
    """Whether one behavioural output is usable on this data, and why."""

    name: str
    status: Status
    reason: str

    @property
    def label(self) -> str:
        return STATUS_LABELS[self.status]

    @property
    def usable(self) -> bool:
        return self.status != "unsupported"


@dataclass
class SufficiencyReport:
    """What the data covers, and what may honestly be concluded from it."""

    detections: int = 0
    cameras: int = 0
    span_months: float = 0.0
    median_per_camera: float = 0.0
    min_per_camera: int = 0
    max_per_camera: int = 0
    first_capture: str = ""
    last_capture: str = ""
    undated: int = 0
    capabilities: list[Capability] = field(default_factory=list)

    def capability(self, name: str) -> Capability | None:
        return next((c for c in self.capabilities if c.name == name), None)

    def supports(self, name: str) -> bool:
        capability = self.capability(name)
        return bool(capability and capability.usable)

    @property
    def overall(self) -> Status:
        if not self.capabilities:
            return "unsupported"
        return _worst(*(c.status for c in self.capabilities))

    def summary_line(self) -> str:
        """One sentence describing the coverage, for a header or a caption."""
        if not self.detections:
            return "No detections to assess."
        return (f"{self.detections:,} detections · {self.cameras} camera"
                f"{'' if self.cameras == 1 else 's'} · "
                f"{_months(self.span_months)} of coverage "
                f"· median {self.median_per_camera:.0f} detections per camera")


def assess(detections: pd.DataFrame,
           timestamp_column: str = "timestamp",
           camera_column: str = "camera_id") -> SufficiencyReport:
    """Assess what a set of detections can support."""
    report = SufficiencyReport()
    if detections is None or detections.empty:
        report.capabilities = [
            Capability(name, "unsupported", "There are no detections to assess.")
            for name in ("activity", "movement", "social")]
        return report

    report.detections = int(len(detections))

    if camera_column in detections.columns:
        per_camera = detections.groupby(camera_column).size()
        report.cameras = int(per_camera.size)
        report.median_per_camera = float(per_camera.median())
        report.min_per_camera = int(per_camera.min())
        report.max_per_camera = int(per_camera.max())

    if timestamp_column in detections.columns:
        # format="mixed" keeps pandas from warning on every ragged timestamp;
        # user-supplied EXIF data is not guaranteed to share one format.
        times = pd.to_datetime(detections[timestamp_column],
                                errors="coerce", format="mixed")
        valid = times.dropna()
        report.undated = int(len(times) - len(valid))
        if not valid.empty:
            report.first_capture = str(valid.min())[:10]
            report.last_capture = str(valid.max())[:10]
            days = (valid.max() - valid.min()).days
            report.span_months = round(days / 30.44, 1)
    else:
        report.undated = report.detections

    report.capabilities = _assess_capabilities(report)
    return report


def _assess_capabilities(report: SufficiencyReport) -> list[Capability]:
    """Grade each behavioural output against what this data provides."""
    capabilities: list[Capability] = []

    # Activity timing — needs detections per camera, not span.
    activity_status = _grade(report.median_per_camera, MIN_DETECTIONS_FOR_ACTIVITY)
    if activity_status == "supported":
        activity_reason = (
            f"The typical camera has {report.median_per_camera:.0f} detections, "
            f"enough to establish whether a species is active by day or night.")
    elif activity_status == "limited":
        activity_reason = (
            f"The typical camera has only {report.median_per_camera:.0f} "
            f"detections. Day/night classification will be decided by a handful "
            f"of photographs and may flip on one or two.")
    else:
        activity_reason = (
            f"The typical camera has {report.median_per_camera:.0f} detections. "
            f"That is too few to say anything about daily activity.")
    capabilities.append(Capability("activity", activity_status, activity_reason))

    # Movement strategy — the demanding one.
    span_status = _grade(report.span_months, MIN_SPAN_MONTHS)
    camera_status = _grade(report.cameras, MIN_CAMERAS)
    detection_status = _grade(report.median_per_camera, MIN_DETECTIONS_PER_CAMERA)
    movement_status = _worst(span_status, camera_status, detection_status)

    if span_status == "unsupported":
        movement_reason = (
            f"The data covers {_months(report.span_months)}. Migratory and "
            f"territorial differ by how presence is spread across seasons, so "
            f"with less than {MIN_SPAN_MONTHS[0]} months every camera looks "
            f"equally concentrated and the classification is meaningless. "
            f"At least {MIN_SPAN_MONTHS[1]} months is needed for a dependable "
            f"answer.")
    elif camera_status == "unsupported":
        movement_reason = (
            f"There are {report.cameras} cameras. Movement classes are assigned "
            f"by comparing each camera against the others, which needs at least "
            f"{MIN_CAMERAS[0]} cameras to mean anything, and "
            f"{MIN_CAMERAS[1]} or more to be dependable.")
    elif detection_status == "unsupported":
        movement_reason = (
            f"The typical camera has only {report.median_per_camera:.0f} "
            f"detections — below the {MIN_DETECTIONS_PER_CAMERA[0]} needed "
            f"before a camera can be considered for the migratory class.")
    elif movement_status == "limited":
        shortfalls = []
        if span_status == "limited":
            shortfalls.append(f"only {_months(report.span_months)} of coverage")
        if camera_status == "limited":
            shortfalls.append(f"only {report.cameras} cameras")
        if detection_status == "limited":
            shortfalls.append(
                f"a median of {report.median_per_camera:.0f} detections per camera")
        movement_reason = (
            "Usable, but read the results as indicative: "
            + " and ".join(shortfalls) + ". Movement classes are relative to "
            "this deployment, so they describe which cameras stand out here, "
            "not absolute animal behaviour.")
    else:
        movement_reason = (
            f"{_months(report.span_months)} across {report.cameras} cameras is "
            f"enough for seasonal contrast to be visible. Note that classes "
            f"remain relative to this deployment.")
    capabilities.append(Capability("movement", movement_status, movement_reason))

    # Social structure — only needs group counts.
    if report.detections == 0:
        social = Capability("social", "unsupported", "No detections.")
    elif report.median_per_camera < 1:
        social = Capability("social", "limited",
                             "Very few detections per camera, so group size is "
                             "based on isolated observations.")
    else:
        social = Capability(
            "social", "supported",
            "Group size is read from the largest number of animals seen in a "
            "single frame, which needs no particular time span.")
    capabilities.append(social)

    return capabilities


def guidance_for(report: SufficiencyReport) -> str:
    """The single most useful next step for improving this data."""
    if not report.detections:
        return ""
    movement = report.capability("movement")
    if movement is None or movement.status == "supported":
        return ""
    if report.span_months < MIN_SPAN_MONTHS[1]:
        return (f"To unlock movement classification, keep the cameras running. "
                f"Coverage of {MIN_SPAN_MONTHS[1]} months or more is what "
                f"matters here — more photographs over the same short window "
                f"will not help.")
    if report.cameras < MIN_CAMERAS[1]:
        return (f"To strengthen movement classification, add cameras. "
                f"{MIN_CAMERAS[1]} or more sites gives the comparison between "
                f"cameras something to work with.")
    return ("To strengthen the results, aim for more detections at the quieter "
            "cameras.")
