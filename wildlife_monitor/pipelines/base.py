"""
Abstract detection pipeline (Package P2).

Defines the common interface and shared workflow for every pipeline. A
concrete pipeline supplies a name, an output directory, and a
:meth:`localise` implementation; the base class handles image loading,
record assembly, overlay saving, and CSV persistence.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from wildlife_monitor.config import RESULTS_DIR, SUBSET_CSV, SystemConfig
from wildlife_monitor.data import load_species_subset
from wildlife_monitor.utils import DetectionRecord, DetectionRepository, save_overlay


class LocalisationResult:
    """The outcome of localising one image."""

    def __init__(
        self,
        kind: str,
        result_image: np.ndarray,
        location: str,
        quality: float,
        mask: np.ndarray | None = None,
        instance_count: int = 1,
        all_masks: list | None = None,
    ) -> None:
        self.kind = kind
        self.result_image = result_image
        self.location = location
        self.quality = quality
        self.mask = mask
        self.instance_count = instance_count
        self.all_masks = all_masks


class DetectionPipeline(ABC):
    """Common workflow for all Pipeline 1 configurations."""

    name: str = "abstract"

    def __init__(self) -> None:
        self.output_dir = RESULTS_DIR / self.name
        self.config = SystemConfig.instance()

    # ── Per-pipeline hooks ─────────────────────────────────────────────────────
    @abstractmethod
    def select_images(self, frame: pd.DataFrame, species: str) -> pd.DataFrame:
        """Return the subset of images this pipeline will process."""

    @abstractmethod
    def localise(
        self, image_bgr: np.ndarray, image_rgb: np.ndarray,
        species: str, confidence: float,
    ) -> LocalisationResult:
        """Localise the target species in one image."""

    # ── Shared driver ──────────────────────────────────────────────────────────
    def run(self, species: str, top_n: int | None = None) -> Path:
        """Run the pipeline for a species and return the output CSV path."""
        top_n = top_n if top_n is not None else self.config.top_n
        self.output_dir.mkdir(parents=True, exist_ok=True)

        frame = load_species_subset(species)
        if frame.empty:
            return self.output_dir / f"detections_{species}.csv"

        selected = self.select_images(frame, species)
        records = self._process(selected, species)

        out_csv = self.output_dir / f"detections_{species}.csv"
        DetectionRepository.save(records, out_csv)
        self._print_summary(out_csv)
        return out_csv

    # ── Internal helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _build_ground_truth_lookup() -> dict[str, str]:
        """Build image_id -> ground_truth_species mapping from subset CSV."""
        if not SUBSET_CSV.exists():
            alt = SUBSET_CSV.parent.parent / "data" / "subset_metadata.csv"
            if not alt.exists():
                return {}
            csv_path = alt
        else:
            csv_path = SUBSET_CSV
        frame = pd.read_csv(csv_path)
        if not {"image_id", "species_label"} <= set(frame.columns):
            return {}
        return dict(zip(
            frame["image_id"].astype(str),
            frame["species_label"].astype(str),
        ))

    def _process(
        self, frame: pd.DataFrame, species: str
    ) -> list[DetectionRecord]:
        ground_truth = self._build_ground_truth_lookup()
        records: list[DetectionRecord] = []

        for _, row in tqdm(frame.iterrows(), total=len(frame),
                           desc=f"{self.name} localising"):
            image_path = Path(row["local_image_path"])
            confidence = float(row.get("confidence", 1.0))

            image_bgr = cv2.imread(str(image_path))
            if image_bgr is None:
                print(f"\n  [WARN] Could not read {image_path.name}; skipping.")
                continue
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            outcome = self.localise(image_bgr, image_rgb, species, confidence)

            stem = image_path.stem
            overlay_path = self.output_dir / "overlays" / f"{stem}_overlay.jpg"
            mask_path = self._save_mask(outcome, stem)

            save_overlay(
                image_bgr, outcome.result_image, overlay_path,
                self.name, species, confidence, outcome.quality,
            )

            image_id_str = str(row.get("image_id", stem))
            gt_species = ground_truth.get(image_id_str, "unknown")
            is_correct = (
                "correct" if gt_species == species
                else "incorrect" if gt_species != "unknown"
                else "unknown"
            )

            records.append(DetectionRecord(
                detection_id=str(uuid.uuid4())[:8],
                image_id=image_id_str,
                pipeline=self.name,
                timestamp=str(row.get("date_captured", "")),
                camera_id=str(row.get("site_id", "")),
                latitude=float(row.get("latitude", 0.0)),
                longitude=float(row.get("longitude", 0.0)),
                habitat_type=str(row.get("habitat_type", "")),
                species=species,
                confidence=round(confidence, 4),
                location_type=outcome.kind,
                location=outcome.location,
                detection_quality=round(outcome.quality, 4),
                instance_count=outcome.instance_count,
                ground_truth_species=gt_species,
                correct=is_correct,
                image_path=str(image_path),
                mask_path=str(mask_path),
                overlay_path=str(overlay_path),
            ))

        return records

    def _save_mask(self, outcome: LocalisationResult, stem: str) -> Path:
        """Persist binary mask PNG(s) when the pipeline produced one or more."""
        if outcome.mask is None:
            return Path("")
        masks_dir = self.output_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), (outcome.mask.astype(np.uint8) * 255))
        # Save all individual instance masks when SAM 3 returned multiple
        if outcome.all_masks and len(outcome.all_masks) > 1:
            for i, m in enumerate(outcome.all_masks):
                inst_path = masks_dir / f"{stem}_instance_{i:02d}.png"
                cv2.imwrite(str(inst_path), (m.astype(np.uint8) * 255))
        return mask_path

    def _print_summary(self, out_csv: Path) -> None:
        print(f"\n[DONE] {self.name} pipeline complete.")
        print(f"       Detections : {out_csv}")
        print(f"       Overlays   : {self.output_dir}/overlays/")
