"""
Pipeline 2 — temporal behavioural analysis (Package P3).

Turns per-camera detection sequences into behavioural classifications:
activity timing, movement strategy and social structure.
"""

from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, SOCIAL_CLASSES,
)
from wildlife_monitor.pipeline2.feature_extractor import TemporalFeatureExtractor
from wildlife_monitor.pipeline2.sequence_builder import (
    SequenceBuilder, DetectionSequence,
)
from wildlife_monitor.pipeline2.models import (
    BehaviourModel, LSTMBehaviourModel, TransformerBehaviourModel,
    build_model, load_checkpoint,
)
from wildlife_monitor.pipeline2.inference import (
    BehaviourService, BehaviourPattern, available_checkpoints,
)
from wildlife_monitor.pipeline2.query import QueryEngine, QueryFilter
from wildlife_monitor.pipeline2.validation import PatternValidator, VERDICTS

__all__ = [
    "ACTIVITY_CLASSES", "MOVEMENT_CLASSES", "SOCIAL_CLASSES",
    "TemporalFeatureExtractor", "SequenceBuilder", "DetectionSequence",
    "BehaviourModel", "LSTMBehaviourModel", "TransformerBehaviourModel",
    "build_model", "load_checkpoint",
    "BehaviourService", "BehaviourPattern", "available_checkpoints",
    "QueryEngine", "QueryFilter", "PatternValidator", "VERDICTS",
]
