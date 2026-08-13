"""
The DetectionRecord data contract and its CSV repository.

DetectionRecord is the single output unit shared by every pipeline. One
record is one detection event and maps directly to one row in a CSV file.
Because all three pipelines emit the same record type, the comparison
runner and the temporal model (Pipeline 2) can consume any pipeline's
output without special-casing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class DetectionRecord:
    """One detection event — the standard output unit of every pipeline.

    The ``location`` field stores either a bounding box as ``"x1,y1,x2,y2"``
    (YOLO pipeline) or the sentinel ``"mask_saved"`` (segmentation
    pipelines), in which case the mask PNG path is held in ``mask_path``.

    Field ordering rule: all fields without defaults must come before all
    fields with defaults. Fields with defaults: mask_path, overlay_path,
    ground_truth_species, correct.
    """

    # Identity
    detection_id: str
    image_id: str
    pipeline: str                 # "bioclip_sam" | "bioclip_yolo" | "sam3"

    # Metadata (carried through from the subset metadata)
    timestamp: str
    camera_id: str
    latitude: float
    longitude: float
    habitat_type: str

    # Detection output
    species: str
    confidence: float             # 0.0 - 1.0
    location_type: str            # "bbox" | "mask"
    location: str                 # bbox coords OR "mask_saved"

    # Quality / evaluation
    detection_quality: float      # SAM quality score OR YOLO confidence

    # File paths (image_path is required; mask and overlay are optional)
    image_path: str

    # Fields with defaults must come last
    mask_path: str = ""           # empty for the YOLO pipeline
    overlay_path: str = ""        # path to the saved visualisation

    # Correctness — derived from ground-truth lookup at pipeline run time.
    # "correct"   — ground truth confirms this image IS the requested species
    # "incorrect" — ground truth says this image is a different species
    # "unknown"   — ground truth not available for this image_id
    ground_truth_species: str = "unknown"
    correct: str = "unknown"


class DetectionRepository:
    """Persists and loads :class:`DetectionRecord` collections as CSV.

    This is the only component that knows how detection records are stored
    on disk, keeping serialisation concerns out of the pipelines.
    """

    @staticmethod
    def save(records: list[DetectionRecord], out_path: Path) -> None:
        """Write a list of records to ``out_path`` as CSV."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            print(f"[WARN] No records to save to {out_path}")
            return
        frame = pd.DataFrame([asdict(record) for record in records])
        frame.to_csv(out_path, index=False)
        print(f"[INFO] Saved {len(records)} detection records -> {out_path}")

    @staticmethod
    def load(csv_path: Path) -> pd.DataFrame:
        """Load a detections CSV, or return an empty frame if it is absent."""
        if not csv_path.exists():
            print(f"[WARN] No detections file found: {csv_path}")
            return pd.DataFrame()
        return pd.read_csv(csv_path)
