"""
SAM 1 segmenter (Package P2).

Wraps the original Segment Anything Model with a five-point cross prompt
centred on the frame. This is the segmenter used by the BioCLIP+SAM
pipeline — the known-good combination that we deliberately keep on SAM 1
rather than upgrading, because BioCLIP already supplies the "what" and
SAM 1 only needs to supply the "where".
"""

from __future__ import annotations

import urllib.request

import numpy as np

from wildlife_monitor.config import (
    SAM1_CHECKPOINT, SAM1_URL, SAM_CROSS_OFFSET, MODELS_DIR, SystemConfig,
)


def ensure_sam1_checkpoint() -> None:
    """Download the SAM 1 ViT-B checkpoint if it is not already present."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if SAM1_CHECKPOINT.exists():
        return
    print("[INFO] Downloading SAM 1 checkpoint (~375 MB) ...")

    def _progress(count: int, block: int, total: int) -> None:
        pct = min(count * block / total * 100, 100)
        print(f"\r       {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(SAM1_URL, SAM1_CHECKPOINT, _progress)
    print()


class SAM1Segmenter:
    """Segments the central subject of an image with a cross-point prompt."""

    def __init__(self) -> None:
        from segment_anything import SamPredictor, sam_model_registry

        ensure_sam1_checkpoint()
        self.device = SystemConfig.instance().device
        print(f"[INFO] Loading SAM 1 predictor on {self.device} ...")
        sam = sam_model_registry["vit_b"](checkpoint=str(SAM1_CHECKPOINT))
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    @staticmethod
    def _cross_points(height: int, width: int) -> np.ndarray:
        """Return five prompt points arranged as a cross about the centre."""
        cx, cy = width // 2, height // 2
        ox, oy = int(width * SAM_CROSS_OFFSET), int(height * SAM_CROSS_OFFSET)
        return np.array([
            [cx, cy],
            [cx - ox, cy], [cx + ox, cy],
            [cx, cy - oy], [cx, cy + oy],
        ])

    def segment(self, image_rgb: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return the best mask and its SAM quality score for one image."""
        self.predictor.set_image(image_rgb)
        points = self._cross_points(*image_rgb.shape[:2])
        labels = np.ones(len(points), dtype=np.int32)
        masks, scores, _ = self.predictor.predict(
            point_coords=points, point_labels=labels, multimask_output=True
        )
        if masks is None or len(masks) == 0:
            return None, 0.0
        best = int(np.argmax(scores))
        return masks[best].astype(bool), float(scores[best])
