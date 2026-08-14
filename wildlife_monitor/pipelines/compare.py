"""
Pipeline comparison runner (Package P2).

Runs all three pipelines on the same species and images, then builds a
metrics report and three-way visual comparisons. Because every pipeline
returns the same :class:`DetectionRecord` structure, the comparator loops
over them uniformly — there is no per-pipeline branching. This directly
answers Research Question 1.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from wildlife_monitor.config import RESULTS_DIR
from wildlife_monitor.pipelines.bioclip_sam import BioCLIPSAMPipeline
from wildlife_monitor.pipelines.bioclip_yolo import BioCLIPYOLOPipeline
from wildlife_monitor.pipelines.megadetector import MegaDetectorPipeline
from wildlife_monitor.utils import DetectionRepository

COMPARE_DIR = RESULTS_DIR / "comparison"

# The pipelines to run, in report column order.
PIPELINES = [BioCLIPSAMPipeline, BioCLIPYOLOPipeline, MegaDetectorPipeline]


class PipelineComparator:
    """Runs and compares the three Pipeline 1 configurations."""

    def __init__(self, species: str, top_n: int) -> None:
        self.species = species
        self.top_n = top_n
        self.timing: dict[str, float] = {}
        self.frames: dict[str, pd.DataFrame] = {}

    def run(self) -> None:
        COMPARE_DIR.mkdir(parents=True, exist_ok=True)
        self._run_all_pipelines()
        self._merge_detections()
        self._build_visual_comparisons()
        self._write_report()
        print(f"\n[DONE] Comparison complete.")
        print(f"       Report  : {COMPARE_DIR / 'comparison_report.txt'}")
        print(f"       Visuals : {COMPARE_DIR / 'comparison_overlays'}/")

    # ── Steps ─────────────────────────────────────────────────────────────────
    def _run_all_pipelines(self) -> None:
        for pipeline_cls in PIPELINES:
            pipeline = pipeline_cls()
            print(f"\n{'=' * 60}")
            print(f"  Running {pipeline.name.upper()} on '{self.species}'")
            print(f"{'=' * 60}")
            start = time.time()
            out_csv = pipeline.run(self.species, self.top_n)
            self.timing[pipeline.name] = time.time() - start
            self.frames[pipeline.name] = DetectionRepository.load(out_csv)

    def _merge_detections(self) -> None:
        present = [df for df in self.frames.values() if not df.empty]
        if not present:
            return
        merged = pd.concat(present, ignore_index=True)
        out_csv = COMPARE_DIR / f"comparison_{self.species}.csv"
        merged.to_csv(out_csv, index=False)
        print(f"\n[INFO] Merged detections -> {out_csv}")

    def _build_visual_comparisons(self) -> None:
        reference = self.frames.get("bioclip_sam", pd.DataFrame())
        if reference.empty:
            return
        print("[INFO] Building three-way visual comparisons ...")
        for _, row in reference.iterrows():
            image_path = str(row["image_path"])
            overlays = self._overlays_for(image_path)
            out_image = (COMPARE_DIR / "comparison_overlays" /
                         f"{Path(image_path).stem}_compare.jpg")
            self._compose_comparison(image_path, overlays, out_image)

    def _overlays_for(self, image_path: str) -> dict[str, str]:
        overlays: dict[str, str] = {}
        for name, frame in self.frames.items():
            if frame.empty:
                continue
            match = frame[frame["image_path"] == image_path]
            if not match.empty:
                overlays[name] = match.iloc[0]["overlay_path"]
        return overlays

    @staticmethod
    def _compose_comparison(
        image_path: str, overlays: dict[str, str], out_path: Path
    ) -> None:
        original = cv2.imread(image_path)
        if original is None:
            return
        height, width = original.shape[:2]

        panels = [original]
        labels = ["Original"]
        for name in ("bioclip_sam", "bioclip_yolo", "bioclip_megadetector"):
            panel = np.zeros_like(original)
            if name in overlays and Path(overlays[name]).exists():
                full = cv2.imread(overlays[name])
                if full is not None:
                    right = full[44 + 24:, full.shape[1] // 2:]
                    panel = cv2.resize(right, (width, height))
            panels.append(panel)
            labels.append(name.replace("_", "+").upper())

        bar = np.ones((36, width * len(panels), 3), dtype=np.uint8) * 25
        cv2.putText(bar, f"Three-pipeline comparison", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        header_strips = []
        for label in labels:
            strip = np.ones((22, width, 3), dtype=np.uint8) * 50
            cv2.putText(strip, label, (6, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
            header_strips.append(strip)

        canvas = np.vstack([bar, np.hstack(header_strips), np.hstack(panels)])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), canvas)

    def _write_report(self) -> None:
        lines = [
            "=" * 66,
            f"Pipeline 1 Comparison Report  |  Species: {self.species}",
            "=" * 66,
            "",
            f"{'Metric':<28} {'BioCLIP+SAM':>13} {'BioCLIP+YOLO':>14} {'SAM 3':>8}",
            "-" * 66,
        ]
        order = ["bioclip_sam", "bioclip_yolo", "bioclip_megadetector"]

        def emit(label: str, value_fn) -> None:
            values = [value_fn(name) for name in order]
            lines.append(f"{label:<28} {values[0]:>13} {values[1]:>14} {values[2]:>8}")

        def processed(name: str) -> str:
            frame = self.frames.get(name, pd.DataFrame())
            return str(len(frame)) if not frame.empty else "0"

        def detected(name: str) -> str:
            frame = self.frames.get(name, pd.DataFrame())
            if frame.empty:
                return "0"
            miss = {"no_mask", "no_detection", ""}
            return str((~frame["location"].isin(miss)).sum())

        def avg_confidence(name: str) -> str:
            frame = self.frames.get(name, pd.DataFrame())
            if frame.empty or "confidence" not in frame:
                return "n/a"
            return f"{frame['confidence'].mean() * 100:.1f}%"

        def avg_quality(name: str) -> str:
            frame = self.frames.get(name, pd.DataFrame())
            if frame.empty or "detection_quality" not in frame:
                return "n/a"
            return f"{frame['detection_quality'].mean():.3f}"

        def runtime(name: str) -> str:
            return f"{self.timing.get(name, 0):.1f}s"

        def per_image(name: str) -> str:
            frame = self.frames.get(name, pd.DataFrame())
            count = len(frame) if not frame.empty else 1
            return f"{self.timing.get(name, 0) / count:.1f}s"

        emit("Images processed", processed)
        emit("Successful detections", detected)
        emit("Avg confidence", avg_confidence)
        emit("Avg detection quality", avg_quality)
        emit("Total runtime", runtime)
        emit("Avg time per image", per_image)

        lines += [
            "=" * 66, "",
            "BioCLIP+SAM  : best spatial precision (pixel mask, SAM 1).",
            "BioCLIP+YOLO : fastest; bounding boxes for large-scale runs.",
            "SAM 3        : text-prompted concept segmentation, single model.",
        ]
        report = "\n".join(lines)
        print("\n" + report)
        (COMPARE_DIR / "comparison_report.txt").write_text(report)
