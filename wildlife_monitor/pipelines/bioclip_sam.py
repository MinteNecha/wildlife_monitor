"""
BioCLIP + SAM 3 pipeline (Pipeline 1a).

BioCLIP ranks images by similarity to the species text prompt, selecting the
top-N most likely images. SAM 3 then performs text-prompted concept
segmentation on each selected image — finding ALL instances of the species
and masking each one individually.

This upgrade from SAM 1 to SAM 3 serves two purposes:
1. Species confirmation — SAM 3 only masks regions that match the species
   concept, rejecting background objects that SAM 1's geometric prompt
   would have incorrectly segmented.
2. Instance counting — SAM 3 returns all detected individuals, providing
   the instance_count that Pipeline 2 uses for social structure
   classification (solitary / small group / large herd).

The best mask (highest SAM 3 confidence) is used for the overlay and stored
as the representative mask. All instance masks are saved individually. The
total count is recorded in DetectionRecord.instance_count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from wildlife_monitor.models import BioCLIPModel, SAM3Segmenter
from wildlife_monitor.pipelines.base import DetectionPipeline, LocalisationResult
from wildlife_monitor.utils import draw_mask


class BioCLIPSAMPipeline(DetectionPipeline):
    """BioCLIP retrieval followed by SAM 3 multi-instance segmentation."""

    name = "bioclip_sam"

    def __init__(self) -> None:
        super().__init__()
        self._segmenter: SAM3Segmenter | None = None

    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Rank images with BioCLIP, then load SAM 3 for localisation."""
        bioclip = BioCLIPModel()
        ranked = bioclip.rank_by_species(frame, species, self.config.top_n)
        bioclip.release()   # free GPU memory before loading SAM 3
        self._segmenter = SAM3Segmenter(species)
        print(f"[INFO] Pipeline 1a backend: {self._segmenter.backend}")
        return ranked

    def localise(
        self,
        image_bgr: np.ndarray,
        image_rgb: np.ndarray,
        species: str,
        confidence: float,
    ) -> LocalisationResult:
        assert self._segmenter is not None

        # SAM 3 returns ALL instances of the species concept
        masks, scores, count = self._segmenter.segment_all(image_rgb)

        if count > 0:
            # Best mask = highest SAM 3 confidence — used for the overlay
            best_idx    = int(np.argmax(scores))
            best_mask   = masks[best_idx]
            best_score  = scores[best_idx]
            result_img  = draw_mask(image_bgr, best_mask, species)
            return LocalisationResult(
                kind="mask",
                result_image=result_img,
                location="mask_saved",
                quality=best_score,
                mask=best_mask,
                instance_count=count,
                all_masks=masks,
            )

        return LocalisationResult(
            kind="mask",
            result_image=image_bgr.copy(),
            location="no_mask",
            quality=0.0,
            mask=None,
            instance_count=0,
        )
