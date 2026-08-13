"""
Central configuration for the Wildlife Monitor system.

This module implements the SystemConfig singleton from the P4 package
of the system design. Every other module reads its paths, model names,
and hyper-parameters from here so that there is exactly one place to
change any setting.

The singleton pattern guarantees that a configuration change made on
one part of the system (for example, the dashboard Settings page) is
immediately visible everywhere else without passing the object around.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch


# ── Project directory layout ──────────────────────────────────────────────────
# All paths are resolved relative to the project root (two levels up
# from this file: wildlife_monitor/wildlife_monitor/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR    = PROJECT_ROOT / "data"
MODELS_DIR  = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

SUBSET_CSV  = DATA_DIR / "subset_metadata.csv"
IMAGES_DIR  = DATA_DIR / "images"

# Allow an explicit override via environment variable. This is the most
# reliable way to point the system at your images regardless of the
# project folder layout:
#   PowerShell:  $env:WM_IMAGES_DIR = "C:\\full\\path\\to\\images"
import os as _os
if _os.environ.get("WM_IMAGES_DIR"):
    IMAGES_DIR = Path(_os.environ["WM_IMAGES_DIR"])
if _os.environ.get("WM_SUBSET_CSV"):
    SUBSET_CSV = Path(_os.environ["WM_SUBSET_CSV"])


# ── Model identifiers and checkpoints ─────────────────────────────────────────
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip"

# SAM 1 (used by the BioCLIP+SAM pipeline — the known-good combination)
SAM1_CHECKPOINT = MODELS_DIR / "sam_vit_b_01ec64.pth"
SAM1_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)

# SAM 3 (used by the standalone text-prompted pipeline — the latest model)
# Weights are gated on Hugging Face and must be downloaded manually; see
# scripts/setup_data.py for instructions. When absent, the pipeline falls
# back to SAM 1 automatically.
SAM3_CHECKPOINT = MODELS_DIR / "sam3.pt"

# YOLOv11 nano — Ultralytics downloads this automatically on first use.
YOLO_MODEL = "yolo11n.pt"

# MegaDetector — Microsoft AI for Earth camera trap animal detector.
# Downloads automatically (~700 MB) on first use via the megadetector package.
# Used by Pipeline 1c (BioCLIP + MegaDetector) for multi-instance counting.
MEGADETECTOR_CONF = 0.20   # minimum confidence for each animal detection


# ── Detection hyper-parameters ────────────────────────────────────────────────
# BioCLIP prompt template. {species} is filled in at runtime.
PROMPT_TEMPLATE = "a camera trap photo of a {species} in African savanna"

# SAM 1 five-point cross prompt offset, as a fraction of image size.
SAM_CROSS_OFFSET = 0.12

# SAM 3 concept-segmentation confidence threshold.
SAM3_CONF = 0.25

# COCO class IDs accepted as "animal" by the YOLO pipeline. Accepting a
# broad set lets YOLO localise Serengeti species without any retraining.
ANIMAL_COCO_IDS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

# Default number of top-ranked images to localise per pipeline run.
DEFAULT_TOP_N = 15


# ── The 13 Serengeti target species ───────────────────────────────────────────
# Canonical labels must match the normalised labels produced by the data
# loader. Used for validation and for the dashboard species selector.
TARGET_SPECIES = [
    "buffalo", "cheetah", "elephant", "giraffe", "leopard",
    "wildebeest", "zebra", "lionmale", "lionfemale", "lioncub",
    "hyenaspotted", "hyenabrown", "gazellethomsons",
]


@dataclass
class SystemConfig:
    """
    Runtime configuration singleton (P4).

    Holds the settings that can change between runs: which device to use,
    the active confidence threshold, and how many images to process.
    Use SystemConfig.instance() to obtain the shared object; never
    construct it directly in application code.
    """

    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    confidence_threshold: float = 0.0
    top_n: int = DEFAULT_TOP_N

    _instance: Optional["SystemConfig"] = field(
        default=None, repr=False, compare=False
    )

    # ── Singleton access ──────────────────────────────────────────────────────
    @classmethod
    def instance(cls) -> "SystemConfig":
        """Return the shared configuration, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls._load_or_default()
        return cls._instance

    @classmethod
    def _load_or_default(cls) -> "SystemConfig":
        config_file = PROJECT_ROOT / "config.json"
        if config_file.exists():
            try:
                data = json.loads(config_file.read_text())
                return cls(
                    device=data.get("device",
                                    "cuda" if torch.cuda.is_available() else "cpu"),
                    confidence_threshold=data.get("confidence_threshold", 0.0),
                    top_n=data.get("top_n", DEFAULT_TOP_N),
                )
            except (json.JSONDecodeError, OSError):
                pass
        return cls()

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self) -> None:
        """Persist the current settings to config.json in the project root."""
        config_file = PROJECT_ROOT / "config.json"
        payload = {
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "top_n": self.top_n,
        }
        config_file.write_text(json.dumps(payload, indent=2))


def ensure_directories() -> None:
    """Create the standard project directories if they do not yet exist."""
    for directory in (DATA_DIR, MODELS_DIR, RESULTS_DIR, IMAGES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
