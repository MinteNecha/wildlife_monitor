"""
BioCLIP + YOLOv11 pipeline (Pipeline 1b).

BioCLIP ranks images by similarity to the species text prompt and selects
the top-N most likely images. YOLOv11 then detects ALL animals in each
selected image and draws a bounding box around every individual.

This upgrade from single-box to multi-box detection makes Pipeline 1b
comparable to Pipeline 1c (MegaDetector) in terms of instance counting,
while remaining the fastest pipeline overall.

Trade-offs vs the other pipelines
-----------------------------------
Pipeline 1a (BioCLIP + SAM 3)    — pixel masks, multi-instance, slowest
Pipeline 1b (BioCLIP + YOLO)     — bounding boxes, multi-instance, fastest
Pipeline 1c (BioCLIP + MegaDetector) — bounding boxes, multi-instance,
                                       camera-trap trained, moderate speed

YOLO is faster than MegaDetector because it is a lighter model (6 MB nano
vs 280 MB MegaDetector). The trade-off is that YOLO was trained on COCO
(80 general classes, 10 animal classes) while MegaDetector was trained
specifically on camera trap imagery and handles all animal sizes and angles
more reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import BioCLIPModel, YOLODetector
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_box


class BioCLIPYOLOPipeline(DetectionPipeline):
    """BioCLIP retrieval followed by YOLOv11 multi-instance detection."""

    name = "bioclip_yolo"

    def __init__(self) -> None:
        super().__init__()
        self._detector: YOLODetector | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Rank images with BioCLIP, then load YOLO."""
        bioclip = BioCLIPModel()
        ranked = bioclip.rank_by_species(frame, species, self.config.top_n)
        bioclip.release()
        self._detector = YOLODetector()
        return ranked

    def localise(
        self,
        image_bgr: np.ndarray,
        image_rgb: np.ndarray,
        species: str,
        confidence: float,
    ) -> LocalisationResult:
        assert self._detector is not None

        # Detect ALL animals in the image above 0.20 confidence
        all_boxes, count = self._detector.detect_all(image_bgr, conf_threshold=0.20)

        if count > 0:
            # Best box = highest YOLO confidence
            best = max(all_boxes, key=lambda b: b[4])
            x1, y1, x2, y2, yolo_conf = best

            # Draw all boxes on the result image
            result_img = image_bgr.copy()
            for bx in all_boxes:
                bx1, by1, bx2, by2, bconf = bx
                result_img = draw_box(
                    result_img, bx1, by1, bx2, by2, species, bconf
                )

            return LocalisationResult(
                kind="bbox",
                result_image=result_img,
                location=f"{x1},{y1},{x2},{y2}",
                quality=yolo_conf,
                mask=None,
                instance_count=count,
            )

        return LocalisationResult(
            kind="bbox",
            result_image=image_bgr.copy(),
            location="no_detection",
            quality=0.0,
            mask=None,
            instance_count=0,
        )
