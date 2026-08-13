"""
BioCLIP + MegaDetector pipeline (Pipeline 1c).

BioCLIP ranks images by species text similarity and selects the top-N most
likely images. MegaDetector then finds ALL animals in each selected image
and draws a bounding box around every individual.

Why MegaDetector over YOLO (Pipeline 1b)
-----------------------------------------
Pipeline 1b (BioCLIP + YOLO) returns ONE bounding box per image — the
highest-confidence detection. This is sufficient for identifying the animal's
location but cannot determine how many animals are present.

MegaDetector was specifically trained on camera trap imagery and returns
EVERY animal in the frame regardless of species. The count of returned boxes
is the instance_count stored in DetectionRecord — the primary input to
Pipeline 2's social structure classification:
  - 1 animal  → solitary
  - 2–5       → small group
  - 6+        → large herd

Trade-offs vs Pipeline 1a (BioCLIP + SAM 3)
--------------------------------------------
MegaDetector returns bounding boxes, not pixel masks. This is less spatially
precise than SAM 3 masks but significantly faster. For social structure
classification Pipeline 2 only needs the count — not the exact pixel
boundaries. MegaDetector is therefore the practical option for large-scale
counting without the 26 s/image overhead of SAM 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import BioCLIPModel, MegaDetector
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_box


class MegaDetectorPipeline(DetectionPipeline):
    """BioCLIP retrieval followed by MegaDetector multi-instance detection."""

    name = "bioclip_megadetector"

    def __init__(self) -> None:
        super().__init__()
        self._detector: MegaDetector | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Rank images with BioCLIP, then load MegaDetector."""
        bioclip = BioCLIPModel()
        ranked = bioclip.rank_by_species(frame, species, self.config.top_n)
        bioclip.release()
        self._detector = MegaDetector()
        print(f"[INFO] Pipeline 1c backend: {self._detector.backend}")
        return ranked

    def localise(
        self,
        image_bgr: np.ndarray,
        image_rgb: np.ndarray,
        species: str,
        confidence: float,
    ) -> LocalisationResult:
        assert self._detector is not None

        # MegaDetector returns ALL animal boxes in the image
        boxes, count = self._detector.detect_all(image_rgb)

        if count > 0:
            # Best box = highest MegaDetector confidence
            best = max(boxes, key=lambda b: b[4])
            x1, y1, x2, y2, det_conf = best
            result_img = image_bgr.copy()

            # Draw all detected animal boxes on the result image
            for bx in boxes:
                bx1, by1, bx2, by2, bconf = bx
                result_img = draw_box(
                    result_img, bx1, by1, bx2, by2, species, bconf
                )

            return LocalisationResult(
                kind="bbox",
                result_image=result_img,
                location=f"{x1},{y1},{x2},{y2}",
                quality=det_conf,
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
