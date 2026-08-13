"""Model package — BioCLIP retrieval and the three localisation backends."""

from wildlife_monitor.models.bioclip import BioCLIPModel
from wildlife_monitor.models.sam1 import SAM1Segmenter, ensure_sam1_checkpoint
from wildlife_monitor.models.sam3 import (
    SAM3Segmenter, concept_prompt_for, sam3_weights_available,
)
from wildlife_monitor.models.yolo import YOLODetector

__all__ = [
    "BioCLIPModel",
    "SAM1Segmenter", "ensure_sam1_checkpoint",
    "SAM3Segmenter", "concept_prompt_for", "sam3_weights_available",
    "YOLODetector",
]
