"""Utility package — data records, persistence, validation, visualisation."""

from wildlife_monitor.utils.records import DetectionRecord, DetectionRepository
from wildlife_monitor.utils.validation import ValidationResult
from wildlife_monitor.utils.visualisation import (
    colour_for, draw_box, draw_mask, save_overlay,
)

__all__ = [
    "DetectionRecord", "DetectionRepository", "ValidationResult",
    "colour_for", "draw_box", "draw_mask", "save_overlay",
]
