"""
Species classification for unlabelled photographs (Package P2).

The three detection pipelines answer *"where is the zebra in this image"* —
they need to be told the species. This one answers *"what is in this image"*,
which is what a user with their own camera trap photographs actually has.

Two stages, in this order, and the order is the point:

    1. MegaDetector asks whether the frame contains an animal at all.
    2. BioCLIP asks which species it is, but only for frames that passed.

Running the detector first is not an optimisation. BioCLIP scores an image
against a closed list of candidates and returns the nearest one, so a
photograph of waving grass — which is most of what a real camera trap
produces — comes back as a confident hyena. The project's own evaluation
found that "a confidence threshold alone cannot be used to filter out bad
predictions because the bad ones are not necessarily low-confidence", so
confidence cannot be relied on to catch it afterwards. An animal detector
answers a different question, independently, and that is what makes it a
usable gate.

The detector, the recogniser and the image loader are all injected rather than
constructed here, so the orchestration can be tested without loading either
model or touching the disk.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import pandas as pd

from wildlife_monitor.config import TARGET_SPECIES
from wildlife_monitor.utils.records import DetectionRecord


class AnimalDetector(Protocol):
    """Anything that can say whether a frame contains animals, and where."""

    def detect_all(self, image_rgb: Any) -> tuple[list, int]:
        ...


class SpeciesRecogniser(Protocol):
    """Anything that can rank candidate species for one image."""

    def classify(self, image_path: str, candidates: list[str],
                 top_k: int = 5, text_features: Any = None
                 ) -> list[tuple[str, float]]:
        ...


@dataclass
class ClassificationReport:
    """What classification did across a batch."""

    images: int = 0
    with_animals: int = 0
    empty_frames: int = 0
    unreadable: int = 0
    classified: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)

    @property
    def empty_share(self) -> float:
        return self.empty_frames / self.images if self.images else 0.0

    def summary_line(self) -> str:
        parts = [f"{self.images:,} images",
                 f"{self.with_animals:,} contained an animal",
                 f"{self.empty_frames:,} empty"]
        if self.unreadable:
            parts.append(f"{self.unreadable:,} unreadable")
        return " · ".join(parts)


class SpeciesClassifier:
    """Identifies the species in each image, gated by an animal detector."""

    name = "bioclip_classify"

    def __init__(self, candidates: list[str] | None = None,
                 detector: AnimalDetector | None = None,
                 recogniser: SpeciesRecogniser | None = None,
                 use_gate: bool = True, top_k: int = 5,
                 image_loader: Callable[[str], Any] | None = None) -> None:
        self.candidates = [s.lower() for s in (candidates or TARGET_SPECIES)]
        self.detector = detector
        self.recogniser = recogniser
        self.use_gate = use_gate
        self.top_k = top_k
        self.image_loader = image_loader or load_rgb

    @property
    def mode(self) -> str:
        """Plain description of what this run is looking for."""
        if len(self.candidates) == 1:
            return f"searching for {self.candidates[0]} only"
        if len(self.candidates) < len(TARGET_SPECIES):
            return (f"choosing between {len(self.candidates)} species: "
                    + ", ".join(self.candidates))
        return f"identifying among all {len(self.candidates)} species"

    def classify_frame(self, frame: pd.DataFrame,
                        path_column: str = "image_path"
                        ) -> tuple[list[DetectionRecord], ClassificationReport]:
        """Classify every image in a frame, returning records and a report."""
        report = ClassificationReport(images=len(frame))
        records: list[DetectionRecord] = []

        for _, row in frame.iterrows():
            record = self._classify_one(row, path_column, report)
            if record is not None:
                records.append(record)
        return records, report

    def _classify_one(self, row, path_column: str,
                      report: ClassificationReport) -> DetectionRecord | None:
        image_path = str(row.get(path_column, ""))

        boxes, count, detector_confidence, readable = self._gate(image_path)
        if not readable:
            # An image that cannot be opened is a different problem from an
            # image with no animal in it, and conflating the two would hide a
            # broken path behind a plausible-looking "empty frame" count.
            report.unreadable += 1
            return None
        if self.use_gate and count == 0:
            report.empty_frames += 1
            return None
        if count > 0:
            report.with_animals += 1

        ranked = self.recogniser.classify(
            image_path, self.candidates, self.top_k) if self.recogniser else []
        if not ranked:
            report.unreadable += 1
            return None

        species, confidence = ranked[0]
        report.classified += 1
        report.species_counts[species] = report.species_counts.get(species, 0) + 1

        location = "no_detection"
        if boxes:
            best = max(boxes, key=lambda box: box[4])
            location = f"{best[0]},{best[1]},{best[2]},{best[3]}"

        return DetectionRecord(
            detection_id=str(uuid.uuid4())[:8],
            image_id=str(row.get("image_id", "")),
            pipeline=self.name,
            timestamp=str(row.get("captured_at", row.get("timestamp", ""))),
            camera_id=str(row.get("camera_id", "")),
            latitude=float(row.get("latitude", 0.0) or 0.0),
            longitude=float(row.get("longitude", 0.0) or 0.0),
            habitat_type=str(row.get("habitat_type", "unknown")),
            species=species,
            confidence=round(float(confidence), 4),
            location_type="bbox",
            location=location,
            detection_quality=round(float(detector_confidence), 4),
            instance_count=max(count, 1),
            image_path=image_path,
            # Nothing here is verified. The user has no labels — that is the
            # premise — so verification stays unknown until a human reviews it.
            ground_truth_species="unknown",
            correct="unknown",
        )

    def _gate(self, image_path: str) -> tuple[list, int, float, bool]:
        """Ask the detector whether this frame contains an animal.

        Returns ``(boxes, count, detector_confidence, readable)``. The last
        flag separates "could not open this file" from "no animal here".
        """
        if not (self.use_gate and self.detector):
            return [], 1, 0.0, True

        image_rgb = self.image_loader(image_path)
        if image_rgb is None:
            return [], 0, 0.0, False

        try:
            boxes, count = self.detector.detect_all(image_rgb)
        except Exception:
            return [], 0, 0.0, False

        best = max((box[4] for box in boxes), default=0.0)
        return boxes, count, best, True


def load_rgb(image_path: str):
    """Read an image as RGB, or None when it cannot be opened."""
    try:
        import cv2
    except ImportError:
        return None
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def build_classifier(candidates: list[str] | None = None,
                     use_gate: bool = True) -> SpeciesClassifier:
    """Construct a classifier with the real models loaded."""
    from wildlife_monitor.models import BioCLIPModel
    from wildlife_monitor.models.megadetector import MegaDetector

    detector = MegaDetector() if use_gate else None
    return SpeciesClassifier(candidates=candidates, detector=detector,
                              recogniser=BioCLIPModel(), use_gate=use_gate)
