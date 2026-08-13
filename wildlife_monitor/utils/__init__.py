"""Utility package — data records, persistence, and visualisation helpers."""

from wildlife_monitor.utils.records import DetectionRecord, DetectionRepository
from wildlife_monitor.utils.visualisation import (
    colour_for, draw_box, draw_mask, save_overlay,
)

__all__ = [
    "DetectionRecord", "DetectionRepository",
    "colour_for", "draw_box", "draw_mask", "save_overlay",
]
