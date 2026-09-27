"""
Loading detections for behavioural training (Package P3).

This sits in the package rather than in a script because two scripts need it:
``train_behaviour.py`` trains one model per species, and
``train_cross_species.py`` pools several species into one model. Duplicating
the loader across both would let them drift apart, and a difference in how
detections are loaded would quietly invalidate any comparison between their
results.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from wildlife_monitor.config import RESULTS_DIR
from wildlife_monitor.db import database_exists, load_detections

DEFAULT_PIPELINE = "bioclip_megadetector"

# The Transformer converges faster and overfits sooner on these sequence
# lengths, so it gets fewer epochs by default.
DEFAULT_EPOCHS = {"lstm": 100, "transformer": 40}

METRICS_DIR = RESULTS_DIR / "behaviour"


def load_species_detections(species: str,
                            pipeline: str = DEFAULT_PIPELINE
                            ) -> tuple[pd.DataFrame, str]:
    """Detections for one species, from the database (the system of record).

    Falls back to a detections CSV when the database has no rows for this
    species, so results archived before the migration still train. Returns the
    frame and a short description of where it came from, which callers print
    so a reader knows which source produced a set of results.
    """
    if database_exists():
        frame = load_detections(species, pipeline)
        if not frame.empty:
            return frame, "database"

    csv_path = RESULTS_DIR / pipeline / f"detections_{species}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path), str(csv_path)

    raise FileNotFoundError(
        f"No detections for '{species}' in the database or at {csv_path}. "
        f"Run Pipeline 1 for this species, or import existing CSVs with "
        f"'python scripts/import_results.py'.")


def gather_species(species_list: list[str],
                   pipeline: str = DEFAULT_PIPELINE,
                   report: Any = print) -> dict[str, pd.DataFrame]:
    """Load every species that has detections, skipping those that do not.

    A missing species is not an error here. Cross-species training is run over
    whatever has been processed so far, and stopping the whole run because one
    of thirteen configured species has no detections would make the script
    unusable in practice.
    """
    sources: dict[str, pd.DataFrame] = {}
    missing: list[str] = []

    for species in species_list:
        try:
            frame, source = load_species_detections(species, pipeline)
        except FileNotFoundError:
            missing.append(species)
            continue
        if frame.empty:
            missing.append(species)
            continue
        sources[species] = frame
        if report:
            report(f"[INFO] {len(frame):>7,} detections  {species}  ({source})")

    if missing and report:
        report(f"[INFO] no detections for: {', '.join(missing)}")
    return sources
