"""
Visualisation helpers shared by all pipelines.

Provides species colour lookup, bounding-box and mask drawing, and the
side-by-side overlay used to review pipeline output. Keeping these here
means every pipeline produces visually consistent output.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Species colour map — BGR tuples for OpenCV.
SPECIES_COLOURS = {
    "buffalo":         (0,   165, 255),
    "cheetah":         (0,   255, 255),
    "elephant":        (180, 180, 180),
    "giraffe":         (0,   200, 0),
    "leopard":         (255, 0,   180),
    "wildebeest":      (255, 80,  0),
    "zebra":           (255, 255, 255),
    "lionmale":        (0,   0,   255),
    "lionfemale":      (0,   0,   200),
    "lioncub":         (0,   0,   150),
    "hyenaspotted":    (255, 0,   255),
    "hyenabrown":      (180, 0,   180),
    "gazellethomsons": (0,   210, 120),
}
DEFAULT_COLOUR = (0, 200, 255)


def colour_for(species: str) -> tuple:
    """Return the BGR colour assigned to a species (or a default)."""
    return SPECIES_COLOURS.get(species.lower(), DEFAULT_COLOUR)


def draw_box(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    species: str,
    confidence: float,
) -> np.ndarray:
    """Return a copy of ``image`` with a labelled bounding box drawn on it."""
    out = image.copy()
    colour = colour_for(species)
    cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
    label = f"{species}  {confidence * 100:.1f}%"
    (text_w, text_h), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    )
    cv2.rectangle(out, (x1, y1 - text_h - 8), (x1 + text_w + 6, y1), colour, -1)
    cv2.putText(out, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def draw_mask(
    image: np.ndarray,
    mask: np.ndarray,
    species: str,
    alpha: float = 0.45,
) -> np.ndarray:
    """Return a copy of ``image`` with a translucent mask and outline."""
    out = image.copy()
    colour = colour_for(species)
    out[mask] = (
        (1 - alpha) * image[mask] + alpha * np.array(colour, dtype=np.float32)
    ).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, colour, 2)
    return out


def save_overlay(
    image_bgr: np.ndarray,
    result_image: np.ndarray,
    out_path: Path,
    pipeline: str,
    species: str,
    confidence: float,
    quality: float,
) -> None:
    """Save a two-panel review image (original | pipeline output).

    A label bar across the top records the pipeline name, species, and the
    confidence and quality scores so an image can be reviewed at a glance.
    """
    height, width = image_bgr.shape[:2]

    bar = np.ones((44, width * 2, 3), dtype=np.uint8) * 30
    caption = (
        f"Pipeline: {pipeline}  |  Species: {species}  |  "
        f"Confidence: {confidence * 100:.1f}%  |  Quality: {quality:.3f}"
    )
    cv2.putText(bar, caption, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)

    def header(text: str) -> np.ndarray:
        strip = np.ones((24, width, 3), dtype=np.uint8) * 55
        cv2.putText(strip, text, (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
        return strip

    headers = np.hstack([header("Original"), header(pipeline)])
    panels = np.hstack([image_bgr, result_image])
    canvas = np.vstack([bar, headers, panels])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
    cv2.imwrite(str(out_path), canvas, params)
