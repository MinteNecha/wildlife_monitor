"""
Dataset acquisition and loading (Package P1).

``load_species_subset`` reads the prepared metadata, filters it to a
single species, and resolves every image to an absolute path on disk so
the pipelines can read it regardless of the directory the command was
launched from. 

"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wildlife_monitor.config import SUBSET_CSV, IMAGES_DIR, PROJECT_ROOT


def _candidate_image_dirs() -> list[Path]:
    """Directories to search for an image, most-specific first."""
    return [
        IMAGES_DIR,                                   # configured / env override
        PROJECT_ROOT / "data" / "images",             # inner package data dir
        PROJECT_ROOT.parent / "data" / "images",      # project-root data dir
    ]


def _resolve_image_path(raw_path: str) -> str:
    """Resolve a CSV image path to an absolute path that exists on disk.

    Tries the path as given (absolute, or relative to the project root),
    then looks for the bare filename in each candidate images directory.
    Returns the original string unchanged if nothing is found so the
    caller's "could not read" warning still fires.
    """
    given = Path(raw_path)

    if given.is_absolute() and given.exists():
        return str(given)

    candidate = PROJECT_ROOT / given
    if candidate.exists():
        return str(candidate)

    filename = given.name
    for directory in _candidate_image_dirs():
        candidate = directory / filename
        if candidate.exists():
            return str(candidate)

    return raw_path


def _find_metadata_csv(csv_path: Path) -> Path:
    """Locate subset_metadata.csv across the common project layouts."""
    candidates = [
        csv_path,
        PROJECT_ROOT / "data" / "subset_metadata.csv",
        PROJECT_ROOT.parent / "data" / "subset_metadata.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Subset metadata not found. Looked in:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPlace subset_metadata.csv in one of these locations, or set "
          "$env:WM_SUBSET_CSV to its full path."
    )


def load_species_subset(species: str, csv_path: Path = SUBSET_CSV) -> pd.DataFrame:
    """Load the subset metadata and filter it to one species.

    Every ``local_image_path`` is rewritten to an absolute path that
    exists on disk. An empty frame is returned, with the available
    species printed, when there is no match.
    """
    csv_path = _find_metadata_csv(csv_path)

    frame = pd.read_csv(csv_path)
    matched = frame[frame["species_label"] == species].copy()

    if not matched.empty:
        matched["local_image_path"] = (
            matched["local_image_path"].astype(str).apply(_resolve_image_path)
        )
        found = matched["local_image_path"].apply(lambda p: Path(p).exists()).sum()
        print(
            f"[INFO] Loaded {len(frame)} rows from {csv_path.name}. "
            f"Filtered to {len(matched)} images of '{species}' "
            f"({found} found on disk)."
        )
        if found == 0:
            print("[WARN] None of the image files could be located.")
            print("       Searched:")
            for directory in _candidate_image_dirs():
                print(f"         {directory}")
            print("       Set $env:WM_IMAGES_DIR to your images folder and retry.")
    else:
        available = sorted(frame["species_label"].unique().tolist())
        print(f"[INFO] Loaded {len(frame)} rows from {csv_path.name}.")
        print(f"[WARN] No images found for species '{species}'.")
        print(f"       Available species: {available}")

    return matched.reset_index(drop=True)


class DatasetLoader:
    """Verifies the prepared Snapshot Serengeti subset is present."""

    def __init__(self, csv_path: Path = SUBSET_CSV) -> None:
        self.csv_path = csv_path

    def prepare(self) -> pd.DataFrame:
        """Ensure the subset metadata exists and return it as a frame."""
        csv_path = _find_metadata_csv(self.csv_path)
        frame = pd.read_csv(csv_path)
        species_counts = frame["species_label"].value_counts().to_dict()
        print(f"[INFO] Subset ready: {len(frame)} images across "
              f"{len(species_counts)} species.")
        return frame
