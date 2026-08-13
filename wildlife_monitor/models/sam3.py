"""
SAM 3 concept segmenter — multi-instance, text-prompted (Package P2).

SAM 3 (Meta, November 2025) accepts a plain-text species concept and finds
every instance of that concept in the image simultaneously. This is the key
upgrade over SAM 1: SAM 3 returns N masks — one per individual animal found
— rather than one mask for the dominant central object.

The instance count (N) is the primary contribution to Pipeline 2's social
structure classification: solitary (1), small group (2–5), large herd (6+).

When SAM 3 weights are absent the segmenter falls back to SAM 1, which
returns a single mask and instance_count=1.
"""

from __future__ import annotations

import numpy as np

from wildlife_monitor.config import SAM3_CHECKPOINT, SAM3_CONF
from wildlife_monitor.models.sam1 import SAM1Segmenter


_CONCEPT_PROMPTS = {
    "gazellethomsons": "thomsons gazelle",
    "lionfemale":      "female lion",
    "lionmale":        "male lion",
    "lioncub":         "lion cub",
    "hyenaspotted":    "spotted hyena",
    "hyenabrown":      "brown hyena",
}


def concept_prompt_for(species: str) -> str:
    return _CONCEPT_PROMPTS.get(species, species)


def sam3_weights_available() -> bool:
    return SAM3_CHECKPOINT.exists()


class SAM3Segmenter:
    """
    Text-prompted concept segmenter — returns ALL detected instances.

    Returns
    -------
    masks  : list[np.ndarray]  — one boolean (H, W) array per detected animal
    scores : list[float]       — one confidence score per mask
    count  : int               — total number of individuals detected

    The caller picks the best mask for the overlay (highest score) but stores
    the count for Pipeline 2.
    """

    def __init__(self, species: str) -> None:
        self.species = species
        self.concept = concept_prompt_for(species)
        self._predictor = None
        self._fallback: SAM1Segmenter | None = None

        if sam3_weights_available():
            try:
                self._load_sam3()
                self.backend = "sam3"
                print(f"[INFO] SAM 3 ready — concept prompt: '{self.concept}'")
                return
            except Exception as exc:
                print(f"[WARN] SAM 3 failed to load ({exc}). "
                      f"Falling back to SAM 1.")

        self._print_sam3_hint()
        self._fallback = SAM1Segmenter()
        self.backend = "sam1_fallback"

    def _load_sam3(self) -> None:
        from ultralytics.models.sam import SAM3SemanticPredictor
        overrides = {
            "conf":    SAM3_CONF,
            "task":    "segment",
            "mode":    "predict",
            "model":   str(SAM3_CHECKPOINT),
            "verbose": False,
            "save":    False,
        }
        self._predictor = SAM3SemanticPredictor(overrides=overrides)

    @staticmethod
    def _print_sam3_hint() -> None:
        print(
            "[INFO] SAM 3 weights not found — using SAM 1 fallback.\n"
            "       To enable SAM 3:\n"
            "         1. pip install -U ultralytics (>= 8.3.237)\n"
            "         2. Request access + download sam3.pt from\n"
            "            https://huggingface.co/facebook/sam3\n"
            "         3. Place sam3.pt in the models/ directory."
        )

    def segment_all(
        self, image_rgb: np.ndarray
    ) -> tuple[list[np.ndarray], list[float], int]:
        """
        Segment ALL instances of the species concept in one image.

        Returns
        -------
        masks  : list of boolean (H, W) arrays, one per detected individual
        scores : confidence score per mask
        count  : number of individuals detected (len(masks))
        """
        if self.backend == "sam1_fallback":
            assert self._fallback is not None
            mask, score = self._fallback.segment(image_rgb)
            if mask is None:
                return [], [], 0
            return [mask], [score], 1

        # SAM 3 multi-instance segmentation
        self._predictor.set_image(image_rgb)
        results = self._predictor(text=[self.concept])
        return self._all_instances(results)

    @staticmethod
    def _all_instances(
        results,
    ) -> tuple[list[np.ndarray], list[float], int]:
        """Extract ALL detected instance masks above the confidence threshold."""
        if results is None:
            return [], [], 0

        result = results[0] if isinstance(results, (list, tuple)) else results
        masks  = getattr(result, "masks", None)
        boxes  = getattr(result, "boxes", None)

        if masks is None or masks.data is None or len(masks.data) == 0:
            return [], [], 0

        mask_array = masks.data.cpu().numpy()   # (N, H, W)

        if boxes is not None and boxes.conf is not None and len(boxes.conf):
            confidences = boxes.conf.cpu().numpy().tolist()
        else:
            confidences = [1.0] * len(mask_array)

        bool_masks = [mask_array[i].astype(bool) for i in range(len(mask_array))]
        return bool_masks, confidences, len(bool_masks)
