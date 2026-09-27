"""
Serving the behavioural models (Package P3).

Training produces a checkpoint; this module turns that checkpoint back into
predictions the dashboard can display. It is the only place that knows where
behaviour checkpoints live and how a raw detections frame becomes a
:class:`BehaviourPattern`.

A note on the social-structure class. The proposal specifies three outputs per
sequence. Activity and movement are predicted by the network; social structure
is currently derived by rule from the largest single-frame instance count and
is labelled as such wherever it is surfaced, rather than being presented as a
model prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from wildlife_monitor.config import MODELS_DIR
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, classify_activity, classify_movement,
)
from wildlife_monitor.pipeline2.models import BehaviourModel, load_checkpoint
from wildlife_monitor.pipeline2.sequence_builder import (
    SequenceBuilder, sequences_to_arrays,
)

BEHAVIOUR_DIR = MODELS_DIR / "behaviour"

# Checkpoint name for the model trained across several species at once. It is
# not a species, so it cannot collide with a real one, and it gives the
# fallback below a fixed place to look.
CROSS_SPECIES_KEY = "_cross_species"
ARCHITECTURE_LABELS = {"lstm": "LSTM", "transformer": "Transformer"}


@dataclass
class BehaviourPattern:
    """One camera's behavioural classification, as stored and displayed."""

    camera_id: str
    species: str
    activity_class: str
    movement_class: str
    social_class: str                 # rule-derived, not a model output
    activity_confidence: float
    movement_confidence: float
    detection_count: int
    sequence_truncated: bool
    peak_month: int
    model_version: str
    start_time: str = ""
    end_time: str = ""
    # The rule-derived labels the model was trained against, kept alongside the
    # prediction so the dashboard can show agreement without recomputing.
    rule_activity_class: str = ""
    rule_movement_class: str = ""

    @property
    def activity_agrees(self) -> bool:
        return self.activity_class == self.rule_activity_class

    @property
    def movement_agrees(self) -> bool:
        return self.movement_class == self.rule_movement_class

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_path(species: str, architecture: str) -> Path:
    """Where the checkpoint for one species/architecture pair lives."""
    return BEHAVIOUR_DIR / f"{species}_{architecture.lower()}.npz"


def available_checkpoints() -> list[dict[str, str]]:
    """Every trained behaviour checkpoint currently on disk."""
    if not BEHAVIOUR_DIR.exists():
        return []
    found = []
    for path in sorted(BEHAVIOUR_DIR.glob("*.npz")):
        stem = path.stem
        architecture = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        species = stem[: -(len(architecture) + 1)] if architecture else stem
        cross = species == CROSS_SPECIES_KEY
        found.append({
            "species": species,
            "label": "all species (cross-species)" if cross else species,
            "cross_species": cross,
            "architecture": architecture,
            "path": str(path),
            "trained_at": datetime.fromtimestamp(
                path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        })
    return found


def species_with_models(architecture: str = "") -> list[str]:
    """Species that have a model of their own, excluding the pooled one."""
    return sorted({entry["species"] for entry in available_checkpoints()
                   if not entry["cross_species"]
                   and (not architecture
                        or entry["architecture"] == architecture.lower())})


def cross_species_available(architecture: str = "lstm") -> bool:
    """Whether a cross-species model has been trained."""
    return checkpoint_path(CROSS_SPECIES_KEY, architecture).exists()


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


class BehaviourService:
    """Loads a trained behavioural model and classifies camera sequences."""

    def __init__(self, model: BehaviourModel, metadata: dict[str, Any]) -> None:
        self.model = model
        self.metadata = metadata
        self.max_length = int(metadata.get("max_length", 40))
        # Set by load_for when a cross-species model stands in for a species
        # that has no model of its own.
        self.applied_to = ""

    @classmethod
    def load(cls, species: str, architecture: str = "lstm") -> "BehaviourService":
        """Load the checkpoint for one species and architecture.

        Raises ``FileNotFoundError`` when that species has no trained model.
        Use :meth:`load_for` to fall back to the cross-species model instead.
        """
        model, metadata = load_checkpoint(checkpoint_path(species, architecture))
        return cls(model, metadata)

    @classmethod
    def load_for(cls, species: str, architecture: str = "lstm"
                 ) -> "BehaviourService":
        """The best available model for a species, preferring its own.

        A model trained on this species is used when one exists, because it has
        seen this animal and will be more accurate. Otherwise the cross-species
        model is used, which has never seen this animal but has learned the
        shape of a detection history from others.

        Without this fallback an ecologist uploading a species the project
        never trained on gets nothing at all from the behavioural page.
        """
        try:
            return cls.load(species, architecture)
        except (FileNotFoundError, OSError):
            service = cls.load(CROSS_SPECIES_KEY, architecture)
            service.applied_to = species
            return service

    @property
    def is_cross_species(self) -> bool:
        """True when this model was trained across several species."""
        return bool(self.metadata.get("cross_species"))

    @property
    def trained_species(self) -> list[str]:
        """Which species this model was trained on."""
        if self.is_cross_species:
            return [str(name) for name in
                    self.metadata.get("trained_species", [])]
        species = str(self.metadata.get("species", ""))
        return [species] if species else []

    def saw_species(self, species: str) -> bool:
        """Whether this model was trained on the named species."""
        wanted = str(species).strip().lower()
        return any(wanted == name.strip().lower()
                   for name in self.trained_species)

    def provenance(self, species: str = "") -> str:
        """One plain sentence saying which model answered, and what that means.

        Shown next to predictions so a reader is never left guessing whether a
        classification came from a model that had seen this animal before.
        """
        target = species or getattr(self, "applied_to", "")
        if not self.is_cross_species:
            return (f"Trained on {self.trained_species[0]} detections."
                    if self.trained_species else "Trained on this species.")

        trained = ", ".join(self.trained_species) or "several species"
        if target and not self.saw_species(target):
            return (f"Trained on {trained}, and applied here to {target}, "
                    f"which it has never seen. Expect lower accuracy than for "
                    f"a species with its own trained model.")
        return f"Trained across {trained}."

    @property
    def model_version(self) -> str:
        """Short provenance string shown next to every prediction."""
        architecture = self.metadata.get("architecture", "")
        label = ARCHITECTURE_LABELS.get(architecture, architecture or "model")
        trained = str(self.metadata.get("trained_at", ""))[:10]
        if self.is_cross_species:
            count = len(self.trained_species)
            species = f"cross-species ({count})" if count else "cross-species"
        else:
            species = self.metadata.get("species", "")
        return f"{label} · {species} · {trained}".strip(" ·")

    def predict(self, detections: pd.DataFrame,
                species: str = "") -> list[BehaviourPattern]:
        """Classify every camera present in a detections frame."""
        if detections.empty:
            return []

        species = species or str(self.metadata.get("species", ""))
        builder = SequenceBuilder.for_model(
            self.max_length, self.model.config().get('num_features', 0))
        sequences = builder.build_all(detections, species)
        if not sequences:
            return []

        vectors, lengths, months = sequences_to_arrays(sequences)
        activity_logits, movement_logits = self.model.forward(
            vectors, lengths, months, training=False)

        activity_probs = _softmax(activity_logits.data)
        movement_probs = _softmax(movement_logits.data)

        stats = builder.camera_statistics(detections)
        grouped = builder.group_by_site(detections)

        patterns = []
        for index, sequence in enumerate(sequences):
            activity_idx = int(np.argmax(activity_probs[index]))
            movement_idx = int(np.argmax(movement_probs[index]))
            camera_df = grouped[sequence.camera_id]
            patterns.append(BehaviourPattern(
                camera_id=sequence.camera_id,
                species=species,
                activity_class=ACTIVITY_CLASSES[activity_idx],
                movement_class=MOVEMENT_CLASSES[movement_idx],
                social_class=sequence.social_label,
                activity_confidence=round(float(activity_probs[index, activity_idx]), 4),
                movement_confidence=round(float(movement_probs[index, movement_idx]), 4),
                detection_count=sequence.detection_count,
                sequence_truncated=sequence.truncated,
                peak_month=int(np.argmax(sequence.month_features)) + 1,
                model_version=self.model_version,
                start_time=sequence.start_time,
                end_time=sequence.end_time,
                rule_activity_class=classify_activity(camera_df),
                rule_movement_class=classify_movement(
                    sequence.camera_id, stats["fidelity"],
                    stats["temporal_concentration"], stats["detection_counts"]),
            ))
        return patterns

    def predict_frame(self, detections: pd.DataFrame,
                      species: str = "") -> pd.DataFrame:
        """Same as :meth:`predict`, returned as a dataframe for display."""
        patterns = self.predict(detections, species)
        if not patterns:
            return pd.DataFrame()
        frame = pd.DataFrame([pattern.to_row() for pattern in patterns])
        frame["activity_agrees"] = (frame["activity_class"]
                                     == frame["rule_activity_class"])
        frame["movement_agrees"] = (frame["movement_class"]
                                     == frame["rule_movement_class"])
        return frame
