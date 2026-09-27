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

Verification is tri-state. A detection is "correct" or "incorrect" only when
a ground-truth label exists for that image; otherwise it is "unverified".
Collapsing unverified into incorrect would report a dashboard full of 0%
accuracy to any user whose images are not pre-labelled, which is every user
outside the Snapshot Serengeti dataset this system was developed against.

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

# Confidence bands used throughout the dashboard. Detections in the lowest
# band are the ones worth a human's attention: without ground truth you cannot
# know which predictions are wrong, but you always know which are uncertain.
CONFIDENCE_BANDS = [
    ("high", 0.75, "Confident"),
    ("medium", 0.50, "Uncertain"),
    ("low", 0.00, "Needs review"),
]


def confidence_band(value: float) -> str:
    """Band name for one confidence score."""
    for name, floor, _ in CONFIDENCE_BANDS:
        if float(value) >= floor:
            return name
    return "low"


def band_label(name: str) -> str:
    """Human-readable label for a confidence band."""
    return next((label for key, _, label in CONFIDENCE_BANDS if key == name), name)


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
        frame["verification"] = frame["correct"].map(
            lambda v: str(v).lower()
            if str(v).lower() in ("correct", "incorrect") else "unverified")
    else:
        # Derive from ground truth (legacy sources without the column)
        ground_truth = _ground_truth_lookup(species)
        labels = frame["image_id"].map(ground_truth)
        frame["verification"] = labels.map(
            lambda label: "unverified" if pd.isna(label)
            else "correct" if label == species else "incorrect")

    # Kept as a convenience for callers that only care about confirmed hits.
    # Never true for unverified detections, and never false either — check
    # ``verification`` when the distinction matters.
    frame["correct"] = frame["verification"].eq("correct")

    if "confidence" in frame.columns:
        frame["band"] = frame["confidence"].map(confidence_band)

    if "location" in frame.columns:
        frame["localised"] = ~frame["location"].isin(_EMPTY_LOCATIONS)
    else:
        frame["localised"] = False
    return frame


def has_ground_truth(frame: pd.DataFrame) -> bool:
    """True when at least one detection in this frame carries a known label."""
    if frame.empty or "verification" not in frame.columns:
        return False
    return bool((frame["verification"] != "unverified").any())


def verification_counts(frame: pd.DataFrame) -> dict[str, int]:
    """How many detections are correct, incorrect and unverified."""
    if frame.empty or "verification" not in frame.columns:
        return {"correct": 0, "incorrect": 0, "unverified": 0}
    counts = frame["verification"].value_counts().to_dict()
    return {key: int(counts.get(key, 0))
            for key in ("correct", "incorrect", "unverified")}


def accuracy_of(frame: pd.DataFrame) -> float | None:
    """Accuracy over verified detections only, or None when none are verified."""
    if frame.empty or "verification" not in frame.columns:
        return None
    verified = frame[frame["verification"] != "unverified"]
    if verified.empty:
        return None
    return float(verified["verification"].eq("correct").mean() * 100)


def sufficiency(species: str) -> "object":
    """Assess what the detections for a species can honestly support."""
    from wildlife_monitor.data.sufficiency import assess
    return assess(behaviour_detections(species))


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
    """Architectures usable for this species, including the pooled model.

    A species with no model of its own can still be classified by the
    cross-species model, so its architectures count here. Without this the
    page would refuse to open for exactly the users the pooled model exists
    to serve.
    """
    from wildlife_monitor.pipeline2.inference import CROSS_SPECIES_KEY
    entries = behaviour_checkpoints()
    own = {entry["architecture"] for entry in entries
           if entry["species"] == species}
    pooled = {entry["architecture"] for entry in entries
              if entry["species"] == CROSS_SPECIES_KEY}
    return sorted(own | pooled)


def has_own_behaviour_model(species: str, architecture: str = "") -> bool:
    """Whether this species has a model trained on it specifically."""
    return any(entry["species"] == species
               and (not architecture
                    or entry["architecture"] == architecture)
               for entry in behaviour_checkpoints())


def load_behaviour_metrics(species: str, architecture: str) -> dict:
    """Training metadata and held-out metrics for the model in use.

    Falls back to the cross-species metrics when this species has no model of
    its own, so the provenance panel describes the model that actually
    produced the predictions on screen.
    """
    path = BEHAVIOUR_RESULTS_DIR / f"{species}_{architecture}_metrics.json"
    if not path.exists():
        pooled = BEHAVIOUR_RESULTS_DIR / f"cross_species_{architecture}_metrics.json"
        path = pooled if pooled.exists() else path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def load_behaviour_service(species: str, architecture: str):
    """Best available model for a species, or None when there is none.

    Prefers a model trained on this species and falls back to the
    cross-species model, so an ecologist working on an untrained species
    still gets classifications.
    """
    from wildlife_monitor.pipeline2.inference import BehaviourService
    try:
        return BehaviourService.load_for(species, architecture)
    except (FileNotFoundError, OSError, ValueError, KeyError):
        return None


def load_cross_species_metrics(architecture: str) -> dict:
    """Leave-one-species-out generalisation results, when they exist."""
    path = BEHAVIOUR_RESULTS_DIR / f"cross_species_{architecture}_metrics.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


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
