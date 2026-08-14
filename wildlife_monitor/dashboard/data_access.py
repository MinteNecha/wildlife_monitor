"""
Dashboard data-access layer.

The dashboard never reads CSVs or touches the filesystem directly. All
data loading, path resolution, and metric derivation happens here, behind
a small set of functions that return plain pandas frames. This keeps the
UI code declarative and means a change to the on-disk format only has to
be reflected in one module.

Correctness is derived, not stored: a detection is "correct" when the
species it was run for matches the ground-truth ``species_label`` for that
image in the subset metadata. This mirrors how the pipelines actually
work — they are run per species, so every record in
``detections_<species>.csv`` is a prediction of that species.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildlife_monitor.config import RESULTS_DIR, SUBSET_CSV
from wildlife_monitor.data import load_species_subset
from wildlife_monitor.pipelines import PIPELINE_REGISTRY

# Human-readable species names for display.
PRETTY_NAMES = {
    "gazellethomsons": "Thomson's Gazelle",
    "hyenaspotted": "Spotted Hyena",
    "hyenabrown": "Brown Hyena",
    "lionmale": "Lion (male)",
    "lionfemale": "Lion (female)",
    "lioncub": "Lion (cub)",
}

# Display metadata for each pipeline, keyed by the pipeline name.
PIPELINE_DISPLAY = {
    "bioclip_sam": {"label": "BioCLIP + SAM", "output": "Segmentation mask"},
    "bioclip_yolo": {"label": "BioCLIP + YOLO", "output": "Bounding box"},
    "bioclip_megadetector": {"label": "SAM 3", "output": "Concept mask"},
}

# Values in the ``location`` column that mean "nothing was localised".
_EMPTY_LOCATIONS = {"no_mask", "no_detection", ""}


def pretty(species: str) -> str:
    """Return a human-readable name for a species label."""
    return PRETTY_NAMES.get(species, str(species).replace("_", " ").title())


def subset_available() -> bool:
    """True when the subset metadata file exists."""
    return SUBSET_CSV.exists()


def load_subset() -> pd.DataFrame:
    """Load the full subset metadata, or an empty frame if it is absent."""
    if not SUBSET_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(SUBSET_CSV)


def species_list() -> list[str]:
    """Return the sorted list of species present in the subset."""
    subset = load_subset()
    if subset.empty or "species_label" not in subset.columns:
        return []
    return sorted(subset["species_label"].unique().tolist())


def detections_path(pipeline: str, species: str) -> Path:
    """Return the CSV path for a pipeline's detections of a species."""
    return RESULTS_DIR / pipeline / f"detections_{species}.csv"


def load_detections(pipeline: str, species: str) -> pd.DataFrame:
    """Load one pipeline's detections for a species, with correctness added.

    If the CSV already contains a ``correct`` column (written by the pipeline
    itself via ground-truth lookup), that column is used directly. Otherwise
    correctness is derived by matching ``image_id`` against the subset
    metadata. Returns an empty frame when the pipeline has not been run.
    """
    path = detections_path(pipeline, species)
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if frame.empty:
        return frame

    # Use the stored correct column if available (new pipeline output)
    if "correct" in frame.columns:
        frame["correct"] = frame["correct"].map(
            lambda v: True if str(v).lower() == "correct"
            else False if str(v).lower() == "incorrect"
            else False
        )
    else:
        # Derive from ground truth (legacy CSVs without the column)
        ground_truth = _ground_truth_lookup(species)
        frame["correct"] = frame["image_id"].map(ground_truth).eq(species)

    if "location" in frame.columns:
        frame["localised"] = ~frame["location"].isin(_EMPTY_LOCATIONS)
    else:
        frame["localised"] = False
    return frame


def load_all_detections(species: str) -> dict[str, pd.DataFrame]:
    """Load every available pipeline's detections for a species.

    Returns a dict keyed by pipeline name; pipelines that have not been run
    are omitted so callers can simply iterate over what is present.
    """
    result: dict[str, pd.DataFrame] = {}
    for pipeline in PIPELINE_REGISTRY:
        frame = load_detections(pipeline, species)
        if not frame.empty:
            result[pipeline] = frame
    return result


def load_comparison_report(species: str) -> str:
    """Return the text of the comparison report, or an empty string."""
    path = RESULTS_DIR / "comparison" / "comparison_report.txt"
    return path.read_text() if path.exists() else ""


def overlay_paths(pipeline: str, limit: int | None = None) -> list[Path]:
    """Return overlay image paths for a pipeline, newest first."""
    directory = RESULTS_DIR / pipeline / "overlays"
    if not directory.exists():
        return []
    paths = sorted(directory.glob("*_overlay.jpg"))
    return paths[:limit] if limit else paths


def _ground_truth_lookup(species: str) -> dict[str, str]:
    """Map image_id -> ground-truth species_label for the whole subset.

    Built from the full subset so a detection's correctness can be checked
    even when the pipeline processed only a ranked sub-selection.
    """
    subset = load_subset()
    if subset.empty or not {"image_id", "species_label"} <= set(subset.columns):
        return {}
    return dict(zip(subset["image_id"].astype(str),
                    subset["species_label"].astype(str)))


# ── Evaluation data access ────────────────────────────────────────────────────

EVAL_DIR = RESULTS_DIR / "evaluation"


def evaluation_available() -> bool:
    """True when the multi-species evaluation results CSV exists."""
    return (EVAL_DIR / "evaluation_results.csv").exists()


def comparison_available() -> bool:
    """True when the BioCLIP vs SAM 3 comparison results CSV exists."""
    return (EVAL_DIR / "bioclip_vs_sam3_results.csv").exists()


def load_evaluation_results() -> pd.DataFrame:
    """Load the multi-species BioCLIP evaluation results."""
    path = EVAL_DIR / "evaluation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_comparison_results() -> pd.DataFrame:
    """Load the BioCLIP vs SAM 3 comparison results."""
    path = EVAL_DIR / "bioclip_vs_sam3_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_evaluation_summary() -> str:
    """Load the BioCLIP vs SAM 3 comparison summary text."""
    path = EVAL_DIR / "bioclip_vs_sam3_summary.txt"
    return path.read_text() if path.exists() else ""


def evaluation_species_list() -> list[str]:
    """Return the species present in the evaluation results."""
    frame = load_evaluation_results()
    if frame.empty or "true_species" not in frame.columns:
        return []
    return sorted(frame["true_species"].unique().tolist())


def per_species_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute per-species accuracy summary from an evaluation frame."""
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for sp, group in frame.groupby("true_species"):
        n = len(group)
        correct_col = (
            "bioclip_correct" if "bioclip_correct" in frame.columns
            else "correct"
        )
        c = int(group[correct_col].sum()) if correct_col in group.columns else 0
        rows.append({
            "species":       sp,
            "display_name":  pretty(sp),
            "images":        n,
            "correct":       c,
            "accuracy_pct":  round(c / n * 100, 1) if n else 0.0,
        })
    return (pd.DataFrame(rows)
              .sort_values("accuracy_pct", ascending=False)
              .reset_index(drop=True))
