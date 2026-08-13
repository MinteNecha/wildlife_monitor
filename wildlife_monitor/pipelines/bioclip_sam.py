"""
BioCLIP + SAM 1 pipeline (Pipeline 1a).

BioCLIP ranks images by similarity to the species prompt, then SAM 1
segments the subject of each top-ranked image with a centre-cross prompt.
This is the reference pipeline that deliberately stays on SAM 1: BioCLIP
supplies species recognition, so the segmenter only needs to supply a
mask, and SAM 1 does that reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import BioCLIPModel, SAM1Segmenter
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_mask


class BioCLIPSAMPipeline(DetectionPipeline):
    """BioCLIP retrieval followed by SAM 1 segmentation."""

    name = "bioclip_sam"

    def __init__(self) -> None:
        super().__init__()
        self._segmenter: SAM1Segmenter | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Rank with BioCLIP, then release it before SAM loads."""
        bioclip = BioCLIPModel()
        ranked = bioclip.rank_by_species(frame, species, self.config.top_n)
        bioclip.release()
        self._segmenter = SAM1Segmenter()
        return ranked

    def localise(
        self, image_bgr: np.ndarray, image_rgb: np.ndarray,
        species: str, confidence: float,
    ) -> LocalisationResult:
        assert self._segmenter is not None
        mask, score = self._segmenter.segment(image_rgb)
        if mask is not None:
            return LocalisationResult(
                kind="mask",
                result_image=draw_mask(image_bgr, mask, species),
                location="mask_saved",
                quality=score,
                mask=mask,
            )
        return LocalisationResult(
            kind="mask", result_image=image_bgr.copy(),
            location="no_mask", quality=0.0, mask=None,
        )
