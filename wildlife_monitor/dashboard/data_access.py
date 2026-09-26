"""
Dashboard data-access layer.

The dashboard never reads CSVs or touches the filesystem directly. All
data loading, path resolution, and metric derivation happens here, behind
a small set of functions that return plain pandas frames. This keeps the
UI code declarative and means a change to the on-disk format only has to
be reflected in one module.

Reads come from the SQLite database, which is the system of record. When the
database has no rows for a species — results archived before the migration, or
a fresh clone — the matching detections CSV is read instead, so the dashboard
still shows whatever the user actually has.

Correctness is derived, not stored: a detection is "correct" when the
species it was run for matches the ground-truth ``species_label`` for that
image in the subset metadata. This mirrors how the pipelines actually
work — they are run per species, so every record in
``detections_<species>.csv`` is a prediction of that species.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wildlife_monitor.config import RESULTS_DIR, SUBSET_CSV
from wildlife_monitor.data import load_species_subset
from wildlife_monitor.db import database_exists, load_detections as db_detections

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
    "bioclip_sam": {
        "label":  "BioCLIP + SAM 3",
        "output": "Multi-instance pixel masks",
    },
    "bioclip_yolo": {
        "label":  "BioCLIP + YOLO",
        "output": "Bounding box (single best)",
    },
    "bioclip_megadetector": {
        "label":  "BioCLIP + MegaDetector",
        "output": "Multi-instance bounding boxes",
    },
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
    """Species available to the dashboard — from the database, else the subset."""
    if database_exists():
        try:
            from wildlife_monitor.db import session, DetectionRepository
            with session() as connection:
                names = DetectionRepository(connection).species_with_detections()
            if names:
                return names
        except Exception:
            pass
    subset = load_subset()
    if subset.empty or "species_label" not in subset.columns:
        return []
    return sorted(subset["species_label"].unique().tolist())


def detections_path(pipeline: str, species: str) -> Path:
    """Return the CSV path for a pipeline's detections of a species."""
    return RESULTS_DIR / pipeline / f"detections_{species}.csv"


def load_raw_detections(pipeline: str, species: str) -> pd.DataFrame:
    """Detections from the database, falling back to the archived CSV."""
    if database_exists():
        frame = db_detections(species, pipeline)
        if not frame.empty:
            return frame
    path = detections_path(pipeline, species)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_detections(pipeline: str, species: str) -> pd.DataFrame:
    """Load one pipeline's detections for a species, with correctness added.

    If the source already contains a ``correct`` column (written by the
    pipeline via ground-truth lookup), that column is used directly. Otherwise
    correctness is derived by matching ``image_id`` against the subset
    metadata. Returns an empty frame when the pipeline has not been run.
    """
    frame = load_raw_detections(pipeline, species)
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
    for pipeline in PIPELINE_DISPLAY:
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
    return (EVAL_DIR / "bioclip_vs_megadetector_results.csv").exists()


def load_evaluation_results() -> pd.DataFrame:
    """Load the multi-species BioCLIP evaluation results."""
    path = EVAL_DIR / "evaluation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_comparison_results() -> pd.DataFrame:
    """Load the BioCLIP vs SAM 3 comparison results."""
    path = EVAL_DIR / "bioclip_vs_megadetector_results.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_evaluation_summary() -> str:
    """Load the BioCLIP vs SAM 3 comparison summary text."""
    path = EVAL_DIR / "bioclip_vs_megadetector_summary.txt"
    return path.read_text() if path.exists() else ""


def evaluation_species_list() -> list[str]:
    """Return the species present in the evaluation results."""
    frame = load_evaluation_results()
    if frame.empty or "true_species" not in frame.columns:
        return []
    return sorted(frame["true_species"].unique().tolist())


# ── Behavioural analysis data access (Pipeline 2 / P3) ───────────────────────

BEHAVIOUR_RESULTS_DIR = RESULTS_DIR / "behaviour"
BEHAVIOUR_PIPELINE = "bioclip_megadetector"


def behaviour_checkpoints() -> list[dict[str, str]]:
    """Every trained behaviour model on disk, newest information first."""
    from wildlife_monitor.pipeline2.inference import available_checkpoints
    return available_checkpoints()


def behaviour_architectures(species: str) -> list[str]:
    """Architectures with a trained checkpoint for this species."""
    return sorted({entry["architecture"] for entry in behaviour_checkpoints()
                   if entry["species"] == species})


def load_behaviour_metrics(species: str, architecture: str) -> dict:
    """Training metadata and held-out metrics for one trained model."""
    path = BEHAVIOUR_RESULTS_DIR / f"{species}_{architecture}_metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_behaviour_service(species: str, architecture: str):
    """Load the trained model for a species, or None when it is absent."""
    from wildlife_monitor.pipeline2.inference import BehaviourService
    try:
        return BehaviourService.load(species, architecture)
    except (FileNotFoundError, ValueError, KeyError):
        return None


def behaviour_detections(species: str) -> pd.DataFrame:
    """The detections frame Pipeline 2 is trained and served on."""
    return load_raw_detections(BEHAVIOUR_PIPELINE, species)


def stored_patterns(species: str, model_version: str | None = None) -> pd.DataFrame:
    """Behaviour patterns persisted by the last training run."""
    if not database_exists():
        return pd.DataFrame()
    try:
        from wildlife_monitor.db import session
        from wildlife_monitor.db.repository import BehaviourPatternRepository
        with session() as connection:
            return BehaviourPatternRepository(connection).frame(
                species, model_version)
    except Exception:
        return pd.DataFrame()


def monthly_profile(detections: pd.DataFrame, camera_id: str) -> pd.DataFrame:
    """Monthly share of one camera's detections — what the model reads."""
    from wildlife_monitor.pipeline2.labelling import compute_monthly_distribution
    distribution = compute_monthly_distribution(detections)
    shares = distribution.get(camera_id)
    if shares is None:
        return pd.DataFrame()
    return pd.DataFrame({
        "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        "share": [float(value) for value in shares],
    })


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
