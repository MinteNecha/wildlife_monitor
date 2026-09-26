"""
Ecologist validation of discovered patterns (Package P3, FR7).

An ecologist reviews a pattern the model produced and records a verdict:

    validated  the pattern matches what is known about this species and site
    novel      plausible but not previously described — worth investigating
    spurious   an artefact; the model is wrong here

Verdicts are appended, never overwritten, so the review history of a pattern
is preserved and a changed opinion is visible as a later entry.

On confidence: the model's own confidence is left untouched. Overwriting it
with a human verdict would destroy the record of what the model actually
predicted, which is the thing an evaluation needs. Instead an *adjusted*
confidence is derived on demand from the model score and the verdict history,
and is always presented as a separate figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wildlife_monitor.db import session
from wildlife_monitor.db.repository import (
    BehaviourPatternRepository, ValidationRepository,
)

VERDICTS = ("validated", "novel", "spurious")

# How much one verdict moves the adjusted confidence toward its target.
_VERDICT_TARGETS = {"validated": 1.0, "spurious": 0.0, "novel": None}
_WEIGHT = 0.5


@dataclass
class VerdictSummary:
    """Aggregated review state for one pattern."""

    pattern_id: int
    model_confidence: float
    adjusted_confidence: float
    verdicts: dict[str, int]
    latest_verdict: str = ""
    latest_notes: str = ""

    @property
    def review_count(self) -> int:
        return sum(self.verdicts.values())

    @property
    def reviewed(self) -> bool:
        return self.review_count > 0


class PatternValidator:
    """Records verdicts on behavioural patterns and derives confidence (FR7)."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = db_path

    def record_verdict(self, pattern_id: int, verdict: str,
                        notes: str = "") -> int:
        """Store one verdict. Raises ValueError on an unknown verdict."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
        with session(self.db_path) as connection:
            return ValidationRepository(connection).record(
                pattern_id, verdict, notes)

    def update_confidence(self, pattern_id: int) -> float:
        """Adjusted confidence for a pattern, given its review history.

        Each *validated* verdict moves the score halfway toward 1.0 and each
        *spurious* verdict halfway toward 0.0, applied in the order the
        verdicts were given. *novel* records interest without asserting
        correctness, so it does not move the score.
        """
        with session(self.db_path) as connection:
            row = connection.execute(
                "SELECT movement_confidence FROM BehaviourPattern "
                "WHERE pattern_id = ?", (pattern_id,)).fetchone()
            if row is None:
                raise KeyError(f"No pattern with id {pattern_id}")
            score = float(row["movement_confidence"] or 0.0)

            verdicts = [entry["verdict"] for entry in connection.execute(
                "SELECT verdict FROM Validation WHERE pattern_id = ? "
                "ORDER BY validation_id", (pattern_id,))]

        for verdict in verdicts:
            target = _VERDICT_TARGETS.get(verdict)
            if target is not None:
                score += (target - score) * _WEIGHT
        return round(min(max(score, 0.0), 1.0), 4)

    def summary(self, pattern_id: int) -> VerdictSummary:
        """Full review state for one pattern."""
        with session(self.db_path) as connection:
            pattern = connection.execute(
                "SELECT movement_confidence FROM BehaviourPattern "
                "WHERE pattern_id = ?", (pattern_id,)).fetchone()
            if pattern is None:
                raise KeyError(f"No pattern with id {pattern_id}")
            rows = connection.execute(
                "SELECT verdict, notes FROM Validation WHERE pattern_id = ? "
                "ORDER BY validation_id", (pattern_id,)).fetchall()

        counts = {verdict: 0 for verdict in VERDICTS}
        for row in rows:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

        return VerdictSummary(
            pattern_id=pattern_id,
            model_confidence=round(float(pattern["movement_confidence"] or 0.0), 4),
            adjusted_confidence=self.update_confidence(pattern_id),
            verdicts=counts,
            latest_verdict=rows[-1]["verdict"] if rows else "",
            latest_notes=rows[-1]["notes"] if rows else "")

    def review_state(self, species: str | None = None) -> pd.DataFrame:
        """Latest verdict per pattern, for showing next to predictions."""
        with session(self.db_path) as connection:
            return ValidationRepository(connection).latest_verdicts(species)

    def counts(self, species: str | None = None) -> dict[str, int]:
        """How many patterns carry each verdict."""
        with session(self.db_path) as connection:
            frame = ValidationRepository(connection).summary(species)
        if frame.empty:
            return {verdict: 0 for verdict in VERDICTS}
        counts = {verdict: 0 for verdict in VERDICTS}
        counts.update(dict(zip(frame["verdict"], frame["n"])))
        return counts

    def pattern_id_for(self, species: str, camera_id: str,
                        model_version: str = "") -> int | None:
        """Find a stored pattern by the camera it describes."""
        with session(self.db_path) as connection:
            frame = BehaviourPatternRepository(connection).frame(
                species, model_version or None)
        if frame.empty:
            return None
        match = frame[frame["camera_id"].astype(str) == str(camera_id)]
        return int(match.iloc[0]["pattern_id"]) if not match.empty else None
