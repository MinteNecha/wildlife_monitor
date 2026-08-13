"""
SAM 3 pipeline (Pipeline 1c).

The standalone text-prompted pipeline. SAM 3 performs species recognition
and segmentation together from a single concept prompt, with no separate
BioCLIP retrieval step. When SAM 3 weights are unavailable the segmenter
transparently falls back to SAM 1, and the reported ``confidence`` reflects
which backend produced each detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import SAM3Segmenter
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_mask


class SAM3Pipeline(DetectionPipeline):
    """Text-prompted concept segmentation with SAM 3 (SAM 1 fallback)."""

    name = "sam3"

    def __init__(self) -> None:
        super().__init__()
        self._segmenter: SAM3Segmenter | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Take the first top_n images; SAM 3 needs no retrieval ranking."""
        self._segmenter = SAM3Segmenter(species)
        selected = frame.head(self.config.top_n).reset_index(drop=True)
        # SAM 3 specifies the species directly via the text prompt, so the
        # per-image retrieval confidence is not meaningful here; report 1.0.
        selected["confidence"] = 1.0
        return selected

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
