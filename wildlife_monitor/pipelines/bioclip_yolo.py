"""
BioCLIP + YOLOv11 pipeline (Pipeline 1b).

Identical BioCLIP retrieval to Pipeline 1a, but localisation is a YOLOv11
bounding box instead of a SAM mask. YOLO produces boxes far faster than
SAM produces masks, so this pipeline is the throughput-oriented option
for large-scale processing where a box is precise enough.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import BioCLIPModel, YOLODetector
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_box


class BioCLIPYOLOPipeline(DetectionPipeline):
    """BioCLIP retrieval followed by YOLOv11 bounding-box detection."""

    name = "bioclip_yolo"

    def __init__(self) -> None:
        super().__init__()
        self._detector: YOLODetector | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        bioclip = BioCLIPModel()
        ranked = bioclip.rank_by_species(frame, species, self.config.top_n)
        bioclip.release()
        self._detector = YOLODetector()
        return ranked

    def localise(
        self, image_bgr: np.ndarray, image_rgb: np.ndarray,
        species: str, confidence: float,
    ) -> LocalisationResult:
        assert self._detector is not None
        detection = self._detector.detect(image_bgr)
        if detection is not None:
            x1, y1, x2, y2, yolo_conf = detection
            return LocalisationResult(
                kind="bbox",
                result_image=draw_box(image_bgr, x1, y1, x2, y2, species, confidence),
                location=f"{x1},{y1},{x2},{y2}",
                quality=yolo_conf,
                mask=None,
            )
        return LocalisationResult(
            kind="bbox", result_image=image_bgr.copy(),
            location="no_detection", quality=0.0, mask=None,
        )
