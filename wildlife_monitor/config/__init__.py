"""Configuration package — central settings and the SystemConfig singleton."""

from wildlife_monitor.config.settings import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
    SUBSET_CSV, IMAGES_DIR,
    BIOCLIP_MODEL, SAM1_CHECKPOINT, SAM1_URL, SAM3_CHECKPOINT,
    YOLO_MODEL, PROMPT_TEMPLATE, SAM_CROSS_OFFSET, SAM3_CONF,
    ANIMAL_COCO_IDS, DEFAULT_TOP_N, TARGET_SPECIES,
    SystemConfig, ensure_directories,
)

__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "RESULTS_DIR",
    "SUBSET_CSV", "IMAGES_DIR",
    "BIOCLIP_MODEL", "SAM1_CHECKPOINT", "SAM1_URL", "SAM3_CHECKPOINT",
    "YOLO_MODEL", "PROMPT_TEMPLATE", "SAM_CROSS_OFFSET", "SAM3_CONF",
    "ANIMAL_COCO_IDS", "DEFAULT_TOP_N", "TARGET_SPECIES",
    "SystemConfig", "ensure_directories",
]
