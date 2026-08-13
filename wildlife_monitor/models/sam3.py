"""
SAM 3 concept segmenter (Package P2).

SAM 3 (Meta, November 2025) introduced Promptable Concept Segmentation:
given a plain text noun phrase it finds and segments every instance of
that concept in an image, with no geometric prompt and no separate
detector. This is exactly what the standalone text-prompted pipeline
wants, so this pipeline is upgraded from the old Grounded-SAM 2 stack to
SAM 3 via the Ultralytics integration.

Availability is handled gracefully. SAM 3 weights (``sam3.pt``) are gated
on Hugging Face and must be downloaded manually; when they are absent this
class transparently falls back to SAM 1 with a centre-cross prompt so the
pipeline always runs. The active backend is reported so results can be
interpreted correctly.
"""

from __future__ import annotations

import numpy as np

from wildlife_monitor.config import SAM3_CHECKPOINT, SAM3_CONF
from wildlife_monitor.models.sam1 import SAM1Segmenter


# Map internal normalised labels to natural-language concept prompts so
# SAM 3 receives phrases it understands ("thomsons gazelle", not
# "gazellethomsons").
_CONCEPT_PROMPTS = {
    "gazellethomsons": "thomsons gazelle",
    "lionfemale": "female lion",
    "lionmale": "male lion",
    "lioncub": "lion cub",
    "hyenaspotted": "spotted hyena",
    "hyenabrown": "brown hyena",
}


def concept_prompt_for(species: str) -> str:
    """Return a natural-language concept prompt for a species label."""
    return _CONCEPT_PROMPTS.get(species, species)


def sam3_weights_available() -> bool:
    """True when the gated SAM 3 checkpoint is present on disk."""
    return SAM3_CHECKPOINT.exists()


class SAM3Segmenter:
    """Text-prompted concept segmenter backed by SAM 3, SAM 1 as fallback.

    When SAM 3 weights are available the segmenter uses
    ``SAM3SemanticPredictor`` with the species concept prompt and returns
    the highest-confidence instance mask. Otherwise it delegates to
    :class:`SAM1Segmenter`. Inspect :pyattr:`backend` to see which path is
    active (``"sam3"`` or ``"sam1_fallback"``).
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
            except Exception as exc:  # pragma: no cover - install dependent
                print(f"[WARN] SAM 3 failed to load ({exc}). "
                      f"Falling back to SAM 1.")

        self._print_sam3_hint()
        self._fallback = SAM1Segmenter()
        self.backend = "sam1_fallback"

    def _load_sam3(self) -> None:
        from ultralytics.models.sam import SAM3SemanticPredictor

        overrides = {
            "conf": SAM3_CONF,
            "task": "segment",
            "mode": "predict",
            "model": str(SAM3_CHECKPOINT),
            "verbose": False,
            "save": False,
        }
        self._predictor = SAM3SemanticPredictor(overrides=overrides)

    @staticmethod
    def _print_sam3_hint() -> None:
        print(
            "[INFO] SAM 3 weights not found — using SAM 1 fallback.\n"
            "       To enable true text-prompted SAM 3 segmentation:\n"
            "         1. pip install -U ultralytics   (>= 8.3.237)\n"
            "         2. Request access + download sam3.pt from\n"
            "            https://huggingface.co/facebook/sam3\n"
            "         3. Place sam3.pt in the models/ directory."
        )

    def segment(self, image_rgb: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return the best mask and its confidence for one image."""
        if self.backend == "sam1_fallback":
            assert self._fallback is not None
            return self._fallback.segment(image_rgb)

        # SAM 3 concept segmentation. The predictor accepts an image and a
        # list of text concepts, returning masks for all matching instances.
        self._predictor.set_image(image_rgb)
        results = self._predictor(text=[self.concept])

        mask, score = self._best_instance(results)
        return mask, score

    @staticmethod
    def _best_instance(results) -> tuple[np.ndarray | None, float]:
        """Extract the highest-confidence instance mask from SAM 3 output."""
        if results is None:
            return None, 0.0

        # Ultralytics returns a Results object (or list of them) carrying
        # .masks (segmentation) and .boxes (with per-instance confidence).
        result = results[0] if isinstance(results, (list, tuple)) else results
        masks = getattr(result, "masks", None)
        boxes = getattr(result, "boxes", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return None, 0.0

        mask_array = masks.data.cpu().numpy()  # (N, H, W)
        if boxes is not None and boxes.conf is not None and len(boxes.conf):
            confidences = boxes.conf.cpu().numpy()
            best = int(np.argmax(confidences))
            score = float(confidences[best])
        else:
            best, score = 0, 1.0

        return mask_array[best].astype(bool), score
