"""
Rolling per-camera classifications up to one statement per species (VL2).

Every accuracy figure elsewhere in this project measures agreement with the
project's own quartile rule. None of them check the system against real
animals. This module produces the output that makes that check possible.

The model predicts per camera, because that is where the data is and because
"is this site a corridor or a home range" is a question no textbook answers.
But published ecology is written about species: lions are nocturnal, giraffes
are diurnal. To compare the two, the camera predictions have to be counted up
into one label per species. That is all this module does.

It deliberately does **not** compare against the literature itself. Which
paper counts as authoritative for a species is a judgement for the ecologist,
not something to hard-code into a system that would then appear to validate
itself.

Three things stop the count being misleading:

* A camera with very few detections votes on almost no evidence, so thin
  cameras are excluded and counted separately rather than silently included.
* A majority of 40% is not a pattern. Where no label is clearly ahead, the
  result is reported as split rather than given a winner.
* The rule labels are summarised alongside the model predictions, so a
  disagreement with published ecology can be traced to the model or to the
  labelling rule.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, SOCIAL_CLASSES,
)

# A camera needs at least this many detections before its activity label is
# worth counting. Below it the day/night majority turns on one or two
# photographs. This matches MIN_DETECTIONS_FOR_ACTIVITY in data.sufficiency.
MIN_DETECTIONS = 5

# The winning label must hold at least this share of the votes, and lead the
# runner-up by at least this margin, before the result is called clear.
CLEAR_SHARE = 0.50
CLEAR_MARGIN = 0.15


@dataclass
class Rollup:
    """How one set of camera votes came out, for one behaviour dimension."""

    dimension: str
    label: str
    share: float
    votes: int
    counts: dict[str, int] = field(default_factory=dict)
    runner_up: str = ""
    runner_up_share: float = 0.0

    @property
    def clear(self) -> bool:
        """Whether one label is far enough ahead to be worth reporting."""
        return (bool(self.label)
                and self.share >= CLEAR_SHARE
                and (self.share - self.runner_up_share) >= CLEAR_MARGIN)

    @property
    def percentage(self) -> str:
        return f"{self.share * 100:.0f}%"

    def summary_line(self) -> str:
        if not self.votes:
            return f"{self.dimension}: no cameras with enough detections"
        if not self.clear:
            return (f"{self.dimension}: split "
                    f"({self.label} {self.percentage}, "
                    f"{self.runner_up} {self.runner_up_share * 100:.0f}%)")
        return f"{self.dimension}: {self.label} ({self.percentage} of cameras)"


@dataclass
class SpeciesProfile:
    """One species' behaviour, counted up from its cameras."""

    species: str
    cameras: int
    cameras_counted: int
    cameras_excluded: int
    detections: int
    activity: Rollup
    movement: Rollup
    social: Rollup
    rule_activity: Rollup | None = None
    rule_movement: Rollup | None = None

    @property
    def model_and_rule_agree(self) -> dict[str, bool]:
        """Whether the model's species-level label matches the rule's.

        A mismatch with published ecology means something different depending
        on this. If the model and the rule agree, the labelling rule is the
        thing to question. If they disagree, the model is.
        """
        agreement = {}
        for name, predicted, derived in (
                ("activity", self.activity, self.rule_activity),
                ("movement", self.movement, self.rule_movement)):
            if derived is not None and predicted.label and derived.label:
                agreement[name] = predicted.label == derived.label
        return agreement

    def summary_lines(self) -> list[str]:
        lines = [self.activity.summary_line(), self.movement.summary_line(),
                 self.social.summary_line()]
        if self.cameras_excluded:
            lines.append(
                f"{self.cameras_excluded} of {self.cameras} cameras excluded "
                f"for having fewer than {MIN_DETECTIONS} detections")
        return lines


def _column(frame: pd.DataFrame, *names: str) -> str:
    """First column present out of several spellings, or an empty string."""
    for name in names:
        if name in frame.columns:
            return name
    return ""


def _as_frame(patterns: Iterable[Any]) -> pd.DataFrame:
    """Accept a frame, a list of BehaviourPattern objects, or dicts."""
    if isinstance(patterns, pd.DataFrame):
        return patterns
    rows = []
    for pattern in patterns:
        if isinstance(pattern, dict):
            rows.append(pattern)
        else:
            rows.append({name: getattr(pattern, name)
                         for name in dir(pattern)
                         if not name.startswith("_")
                         and not callable(getattr(pattern, name))})
    return pd.DataFrame(rows)


def roll_up(frame: pd.DataFrame, column: str, dimension: str,
            classes: list[str]) -> Rollup:
    """Count one column's votes and report the winner and its margin.

    Only labels in ``classes`` are counted, so a stray or empty value cannot
    win by appearing more often than the real ones.
    """
    if not column or column not in frame.columns or frame.empty:
        return Rollup(dimension=dimension, label="", share=0.0, votes=0)

    valid = [str(value) for value in frame[column]
             if str(value) in classes]
    if not valid:
        return Rollup(dimension=dimension, label="", share=0.0, votes=0)

    counts = Counter(valid)
    # Sort by count, then by the order in classes, so ties resolve the same
    # way on every run rather than by dictionary insertion order.
    ordered = sorted(counts.items(),
                     key=lambda item: (-item[1], classes.index(item[0])))
    total = len(valid)

    label, count = ordered[0]
    runner_up, runner_count = ordered[1] if len(ordered) > 1 else ("", 0)

    return Rollup(
        dimension=dimension,
        label=label,
        share=count / total,
        votes=total,
        counts={name: counts.get(name, 0) for name in classes},
        runner_up=runner_up,
        runner_up_share=runner_count / total if total else 0.0)


def profile_species(patterns, species: str = "",
                    min_detections: int = MIN_DETECTIONS) -> SpeciesProfile:
    """Count one species' camera classifications into a single profile."""
    frame = _as_frame(patterns)
    species_column = _column(frame, "species")
    if species and species_column:
        frame = frame[frame[species_column].astype(str) == species]

    total_cameras = len(frame)
    count_column = _column(frame, "detection_count", "detections")
    if count_column and total_cameras:
        counted = frame[pd.to_numeric(frame[count_column], errors="coerce")
                        .fillna(0) >= min_detections]
    else:
        counted = frame

    detections = 0
    if count_column and total_cameras:
        detections = int(pd.to_numeric(frame[count_column], errors="coerce")
                         .fillna(0).sum())

    rule_activity_column = _column(counted, "rule_activity_class")
    rule_movement_column = _column(counted, "rule_movement_class")

    return SpeciesProfile(
        species=species or "all",
        cameras=total_cameras,
        cameras_counted=len(counted),
        cameras_excluded=total_cameras - len(counted),
        detections=detections,
        activity=roll_up(counted, _column(counted, "activity_class", "activity"),
                         "Activity", ACTIVITY_CLASSES),
        movement=roll_up(counted, _column(counted, "movement_class", "movement"),
                         "Movement", MOVEMENT_CLASSES),
        social=roll_up(counted, _column(counted, "social_class", "social"),
                       "Group size", SOCIAL_CLASSES),
        rule_activity=(roll_up(counted, rule_activity_column, "Activity (rule)",
                               ACTIVITY_CLASSES)
                       if rule_activity_column else None),
        rule_movement=(roll_up(counted, rule_movement_column, "Movement (rule)",
                               MOVEMENT_CLASSES)
                       if rule_movement_column else None),
    )


def profile_all(patterns, min_detections: int = MIN_DETECTIONS
                ) -> list[SpeciesProfile]:
    """One profile per species present, ordered by species name."""
    frame = _as_frame(patterns)
    species_column = _column(frame, "species")
    if frame.empty or not species_column:
        return []

    names = sorted({str(value) for value in frame[species_column]
                    if str(value).strip()})
    return [profile_species(frame, name, min_detections) for name in names]


def to_frame(profiles: list[SpeciesProfile]) -> pd.DataFrame:
    """The species summary as a table, for export and for the dashboard.

    One row per species, with the label, its share, and whether the share was
    clear enough to call. An ecologist compares the label columns against
    published ecology; the share and clarity columns say how much weight each
    one deserves.
    """
    rows = []
    for profile in profiles:
        row = {
            "species": profile.species,
            "cameras": profile.cameras,
            "cameras_counted": profile.cameras_counted,
            "detections": profile.detections,
            "activity": profile.activity.label,
            "activity_agreement": profile.activity.percentage,
            "activity_clear": profile.activity.clear,
            "movement": profile.movement.label,
            "movement_agreement": profile.movement.percentage,
            "movement_clear": profile.movement.clear,
            "group_size": profile.social.label,
            "group_size_agreement": profile.social.percentage,
        }
        if profile.rule_activity is not None:
            row["activity_rule"] = profile.rule_activity.label
        if profile.rule_movement is not None:
            row["movement_rule"] = profile.rule_movement.label
        agreement = profile.model_and_rule_agree
        if agreement:
            row["model_matches_rule"] = all(agreement.values())
        rows.append(row)
    return pd.DataFrame(rows)


def counts_frame(profiles: list[SpeciesProfile]) -> pd.DataFrame:
    """Every class's camera count per species, for the full picture.

    The summary reports only the winner. This shows the whole vote, which is
    what a reader needs to judge whether a 52% majority means anything.
    """
    rows = []
    for profile in profiles:
        for rollup in (profile.activity, profile.movement, profile.social):
            for label, count in rollup.counts.items():
                rows.append({
                    "species": profile.species,
                    "dimension": rollup.dimension,
                    "class": label,
                    "cameras": count,
                    "share": (f"{count / rollup.votes * 100:.0f}%"
                              if rollup.votes else "0%"),
                })
    return pd.DataFrame(rows)
