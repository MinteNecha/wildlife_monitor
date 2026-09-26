"""
Natural-language querying of behavioural patterns (Package P3, FR6).

Turns a phrase like *"nocturnal zebra at kopje sites with more than 50
detections"* into a structured :class:`QueryFilter`, then applies that filter
to stored patterns.

The parser is deliberately rule-based rather than statistical. The project's
learning guide asks for core functionality to be built from the ground up, and
a closed domain — nine behavioural classes, thirteen species, twelve months, a
handful of habitats — is exactly the case where matching against a known
vocabulary is both tractable and inspectable.

Because a rule-based parser will always miss some phrasings, it never fails
silently. :meth:`QueryEngine.parse` records which terms it recognised and
which words it ignored, and the caller is expected to show that back to the
user so they can see what was actually understood.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from wildlife_monitor.config import TARGET_SPECIES
from wildlife_monitor.pipeline2.feature_extractor import HABITAT_CLASSES
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, SOCIAL_CLASSES,
)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Phrases that mean the same as a class name, so ordinary English works.
SYNONYMS = {
    "diurnal": ["daytime", "daylight", "day-active", "during the day"],
    "nocturnal": ["night", "night-time", "nighttime", "night-active", "at night"],
    "crepuscular": ["dawn", "dusk", "twilight", "dawn and dusk"],
    "migratory": ["migrating", "migration", "passing through", "passage",
                  "seasonal"],
    "territorial": ["resident", "site-faithful", "site faithful", "settled"],
    "nomadic": ["wandering", "roaming", "transient"],
    "solitary": ["alone", "single", "lone"],
    "small group": ["small groups", "pairs", "small herd"],
    "large herd": ["large herds", "herd", "herds", "big group", "large group"],
}

STOPWORDS = {
    "show", "me", "find", "list", "all", "the", "a", "an", "of", "in", "at",
    "on", "with", "and", "or", "for", "which", "that", "where", "are", "is",
    "was", "were", "have", "has", "had", "cameras", "camera", "sites", "site",
    "detections", "detection", "species", "please", "any", "more", "than",
    "over", "above", "under", "below", "least", "most", "seen", "during",
    "behaviour", "behavior", "patterns", "pattern", "by", "to", "from",
}


@dataclass
class QueryFilter:
    """A parsed query: every field is optional, unset fields do not filter."""

    species: str | None = None
    activity: str | None = None
    movement: str | None = None
    social: str | None = None
    habitat: str | None = None
    camera_id: str | None = None
    month: int | None = None
    min_detections: int | None = None
    min_confidence: float | None = None
    limit: int | None = None

    raw_query: str = ""
    matched_terms: list[str] = field(default_factory=list)
    unrecognised: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True when nothing in the query constrained the results."""
        return not any([self.species, self.activity, self.movement, self.social,
                        self.habitat, self.camera_id, self.month,
                        self.min_detections, self.min_confidence])

    def describe(self) -> str:
        """Plain-English account of what the parser understood."""
        parts = []
        if self.species:
            parts.append(f"species is {self.species}")
        if self.activity:
            parts.append(f"activity is {self.activity}")
        if self.movement:
            parts.append(f"movement is {self.movement}")
        if self.social:
            parts.append(f"social structure is {self.social}")
        if self.habitat:
            parts.append(f"habitat is {self.habitat}")
        if self.camera_id:
            parts.append(f"camera is {self.camera_id}")
        if self.month:
            month_name = next((name.title() for name, number in MONTHS.items()
                                if number == self.month and len(name) > 3),
                               str(self.month))
            parts.append(f"busiest month is {month_name}")
        if self.min_detections is not None:
            parts.append(f"at least {self.min_detections} detections")
        if self.min_confidence is not None:
            parts.append(f"confidence at least {self.min_confidence:g}")
        if not parts:
            return "no filters recognised — showing everything"
        return "showing patterns where " + ", and ".join(parts)


class QueryEngine:
    """Parses behaviour queries and runs them against stored patterns (FR6)."""

    def __init__(self, species_vocabulary: list[str] | None = None) -> None:
        self.species_vocabulary = [s.lower() for s in
                                    (species_vocabulary or TARGET_SPECIES)]

    # ── Parsing ───────────────────────────────────────────────────────────────
    def parse(self, query: str) -> QueryFilter:
        """Turn a phrase into a structured filter."""
        text = " " + re.sub(r"\s+", " ", str(query).strip().lower()) + " "
        parsed = QueryFilter(raw_query=str(query).strip())
        consumed: set[str] = set()

        parsed.activity = self._match_class(text, ACTIVITY_CLASSES, consumed)
        parsed.movement = self._match_class(text, MOVEMENT_CLASSES, consumed)
        parsed.social = self._match_class(text, SOCIAL_CLASSES, consumed)
        parsed.species = self._match_species(text, consumed)
        parsed.habitat = self._match_habitat(text, consumed)
        parsed.camera_id = self._match_camera(text, consumed)
        parsed.month = self._match_month(text, consumed)
        parsed.min_detections = self._match_min_detections(text)
        if parsed.min_detections is not None:
            consumed.update({"detections", "detection", "sightings"})
        parsed.min_confidence = self._match_min_confidence(text)
        if parsed.min_confidence is not None:
            consumed.add("confidence")
        parsed.limit = self._match_limit(text)
        if parsed.limit is not None:
            consumed.update({"top", "first"})

        parsed.matched_terms = sorted(consumed)
        parsed.unrecognised = [
            word for word in re.findall(r"[a-z][a-z'-]+", text)
            if word not in STOPWORDS
            and not any(word in term or term in word for term in consumed)
        ]
        return parsed

    def _match_class(self, text: str, classes: list[str],
                     consumed: set[str]) -> str | None:
        """Match a behavioural class by name or by an everyday synonym."""
        for class_name in classes:
            candidates = [class_name, *SYNONYMS.get(class_name, [])]
            for candidate in candidates:
                if f" {candidate} " in text:
                    consumed.add(candidate)
                    return class_name
                if f" {candidate}s " in text:
                    consumed.update({candidate, f"{candidate}s",
                                     candidate.split()[-1] + "s"})
                    return class_name
        return None

    def _match_species(self, text: str, consumed: set[str]) -> str | None:
        for species in sorted(self.species_vocabulary, key=len, reverse=True):
            if f" {species} " in text or f" {species}s " in text:
                consumed.add(species)
                return species
        # "thomson's gazelle" and similar written-out forms
        for species in self.species_vocabulary:
            spaced = re.sub(r"(hyena|lion|gazelle)", r"\1 ", species).strip()
            if spaced != species and f" {spaced} " in text:
                consumed.add(spaced)
                return species
        return None

    def _match_habitat(self, text: str, consumed: set[str]) -> str | None:
        for habitat in HABITAT_CLASSES:
            spoken = habitat.replace("_", " ")
            if f" {habitat} " in text or f" {spoken} " in text:
                consumed.add(spoken)
                return habitat
        return None

    @staticmethod
    def _match_camera(text: str, consumed: set[str]) -> str | None:
        """Camera codes such as B04, S1_D02 — a letter-digit site pattern."""
        match = re.search(r"\b([a-z]\d{1,2}[_-]?[a-z]?\d{0,2})\b", text)
        if not match:
            return None
        candidate = match.group(1)
        if not re.search(r"\d", candidate) or len(candidate) < 2:
            return None
        consumed.add(candidate)
        return candidate.upper()

    @staticmethod
    def _match_month(text: str, consumed: set[str]) -> int | None:
        for name, number in MONTHS.items():
            if f" {name} " in text:
                consumed.add(name)
                return number
        return None

    @staticmethod
    def _match_min_detections(text: str) -> int | None:
        match = re.search(
            r"(?:more than|over|at least|above|minimum(?: of)?)\s+(\d+)\s*"
            r"(?:detections?|images?|records?|sightings?)", text)
        if match:
            return int(match.group(1))
        match = re.search(r"(\d+)\s*\+?\s*(?:detections?|sightings?)", text)
        return int(match.group(1)) if match else None

    @staticmethod
    def _match_min_confidence(text: str) -> float | None:
        match = re.search(
            r"confiden\w*\s*(?:of|above|over|at least|>=?|greater than)?\s*"
            r"(\d*\.?\d+)\s*(%?)", text)
        if not match:
            return None
        value = float(match.group(1))
        if match.group(2) == "%" or value > 1:
            value /= 100.0
        return min(max(value, 0.0), 1.0)

    @staticmethod
    def _match_limit(text: str) -> int | None:
        match = re.search(r"(?:top|first)\s+(\d+)", text)
        return int(match.group(1)) if match else None

    # ── Execution ─────────────────────────────────────────────────────────────
    def execute(self, query_filter: QueryFilter,
                patterns: pd.DataFrame) -> pd.DataFrame:
        """Apply a parsed filter to a frame of behaviour patterns."""
        if patterns.empty:
            return patterns

        result = patterns
        for column, value in [
            ("species", query_filter.species),
            ("activity_class", query_filter.activity),
            ("movement_class", query_filter.movement),
            ("social_class", query_filter.social),
            ("habitat_type", query_filter.habitat),
        ]:
            if value and column in result.columns:
                result = result[result[column].astype(str).str.lower()
                                == str(value).lower()]

        if query_filter.month is not None and "peak_month" in result.columns:
            result = result[result["peak_month"] == query_filter.month]

        if query_filter.camera_id and "camera_id" in result.columns:
            result = result[result["camera_id"].astype(str).str.upper()
                            == query_filter.camera_id.upper()]

        if (query_filter.min_detections is not None
                and "detection_count" in result.columns):
            result = result[result["detection_count"]
                            >= query_filter.min_detections]

        if query_filter.min_confidence is not None:
            columns = [c for c in ("movement_confidence", "activity_confidence")
                       if c in result.columns]
            for column in columns[:1]:
                result = result[result[column] >= query_filter.min_confidence]

        if query_filter.limit:
            result = result.head(query_filter.limit)
        return result

    def run(self, query: str,
            patterns: pd.DataFrame) -> tuple[pd.DataFrame, QueryFilter]:
        """Parse and execute in one call, returning the results and the filter."""
        query_filter = self.parse(query)
        return self.execute(query_filter, patterns), query_filter


EXAMPLE_QUERIES = [
    "nocturnal zebra with more than 50 detections",
    "migratory patterns in woodland",
    "territorial cameras with confidence above 0.8",
    "large herds at open grassland",
    "top 10 nomadic cameras",
]
