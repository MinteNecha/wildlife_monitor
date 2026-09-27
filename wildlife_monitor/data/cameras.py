"""
The camera registry (Package P1).

External users have no Snapshot Serengeti metadata file. What they do have is
knowledge of where their own cameras are, so the system asks for exactly that
and nothing else:

    camera_id, latitude, longitude, habitat_type

Two ways in, because both are reasonable. Type the details into a form and the
system writes the file, which removes any chance of getting the format wrong;
or upload a file already prepared, for anyone who keeps this in a spreadsheet.

Either way the entries are validated. The mistakes that actually happen are
worth naming: latitude and longitude swapped (which puts a Serengeti camera in
the Indian Ocean), a camera named in the file that matches no folder of
images, and folders of images with no entry in the file at all. The last is
the quiet one — those images would ingest with no location.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wildlife_monitor.pipeline2.feature_extractor import HABITAT_CLASSES

COLUMNS = ["camera_id", "latitude", "longitude", "habitat_type"]


@dataclass
class Issue:
    """One problem found in a camera file."""

    level: str          # "error" blocks ingestion; "warning" does not
    message: str
    camera_id: str = ""

    @property
    def blocking(self) -> bool:
        return self.level == "error"


class CameraRegistry:
    """Builds, loads and validates the camera metadata file."""

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.frame = frame if frame is not None else self.empty()

    # ── Construction ──────────────────────────────────────────────────────────
    @staticmethod
    def empty() -> pd.DataFrame:
        return pd.DataFrame(columns=COLUMNS)

    @classmethod
    def template(cls, camera_ids: list[str],
                 habitat: str = "unknown") -> pd.DataFrame:
        """A pre-filled skeleton for the cameras actually found on disk.

        The folder names are already known by the time this is needed, so the
        user should never have to type a camera name — only its coordinates.
        """
        return pd.DataFrame([
            {"camera_id": str(camera_id), "latitude": None, "longitude": None,
             "habitat_type": habitat}
            for camera_id in camera_ids])

    @classmethod
    def from_entries(cls, entries) -> "CameraRegistry":
        """Build from typed rows — a list of dicts or a dataframe."""
        frame = (entries.copy() if isinstance(entries, pd.DataFrame)
                 else pd.DataFrame(list(entries)))
        for column in COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return cls(frame[COLUMNS])

    @classmethod
    def load(cls, path: str | Path) -> "CameraRegistry":
        """Load a user-supplied camera file, tolerating extra columns."""
        frame = pd.read_csv(path)
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        # Accept a few obvious alternative spellings before giving up.
        renames = {"camera": "camera_id", "site": "camera_id",
                   "site_id": "camera_id", "lat": "latitude",
                   "lon": "longitude", "lng": "longitude",
                   "habitat": "habitat_type"}
        frame = frame.rename(columns={k: v for k, v in renames.items()
                                       if k in frame.columns
                                       and v not in frame.columns})
        for column in COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        return cls(frame)

    # ── Validation ────────────────────────────────────────────────────────────
    def validate(self, known_cameras: list[str] | None = None) -> list[Issue]:
        """Check the entries, optionally against the folders found on disk."""
        issues: list[Issue] = []
        frame = self.frame

        if frame.empty:
            return [Issue("error", "No camera entries were provided.")]

        missing = [column for column in COLUMNS if column not in frame.columns]
        if missing:
            return [Issue("error", f"Missing columns: {', '.join(missing)}")]

        identifiers = frame["camera_id"].astype(str).str.strip()

        if (identifiers == "").any() or frame["camera_id"].isna().any():
            issues.append(Issue("error", "Every row needs a camera name."))

        duplicates = identifiers[identifiers.duplicated() & (identifiers != "")]
        for camera_id in duplicates.unique():
            issues.append(Issue(
                "error", f"'{camera_id}' appears more than once.", camera_id))

        for _, row in frame.iterrows():
            camera_id = str(row["camera_id"]).strip()
            issues.extend(self._check_coordinates(row, camera_id))
            issues.extend(self._check_habitat(row, camera_id))

        if known_cameras is not None:
            issues.extend(self._cross_check(identifiers, known_cameras))
        return issues

    @staticmethod
    def _check_coordinates(row, camera_id: str) -> list[Issue]:
        issues: list[Issue] = []
        import math

        try:
            latitude = float(row["latitude"])
            longitude = float(row["longitude"])
        except (TypeError, ValueError):
            return [Issue("error",
                           f"'{camera_id}' needs a numeric latitude and "
                           f"longitude.", camera_id)]

        # A blank cell reads as NaN once pandas has loaded the file, which is
        # what a freshly generated template looks like. Say that plainly
        # rather than reporting "nan is outside -90 to 90".
        if math.isnan(latitude) or math.isnan(longitude):
            return [Issue("error",
                           f"'{camera_id}' has no coordinates yet — fill in "
                           f"its latitude and longitude.", camera_id)]

        if not -90 <= latitude <= 90:
            issues.append(Issue(
                "error", f"'{camera_id}' latitude {latitude} is outside "
                          f"-90 to 90.", camera_id))
        if not -180 <= longitude <= 180:
            issues.append(Issue(
                "error", f"'{camera_id}' longitude {longitude} is outside "
                          f"-180 to 180.", camera_id))

        # Latitude and longitude swapped is the classic data-entry error and
        # is silently plausible, so it is worth naming when it is detectable.
        if -90 <= latitude <= 90 and abs(longitude) <= 90 and abs(latitude) > 90:
            issues.append(Issue(
                "warning", f"'{camera_id}' may have latitude and longitude "
                            f"swapped.", camera_id))
        return issues

    @staticmethod
    def _check_habitat(row, camera_id: str) -> list[Issue]:
        habitat = str(row.get("habitat_type") or "unknown").strip().lower()
        if habitat and habitat not in HABITAT_CLASSES:
            return [Issue(
                "warning",
                f"'{camera_id}' habitat '{habitat}' is not one of "
                f"{', '.join(HABITAT_CLASSES)} and will be treated as "
                f"unknown.", camera_id)]
        return []

    @staticmethod
    def _cross_check(identifiers: pd.Series,
                     known_cameras: list[str]) -> list[Issue]:
        """Compare the file against the camera folders actually found."""
        issues: list[Issue] = []
        listed = {value for value in identifiers if value}
        found = {str(name) for name in known_cameras}

        for camera_id in sorted(listed - found):
            issues.append(Issue(
                "warning", f"'{camera_id}' is in the camera file but no folder "
                            f"of images was found for it.", camera_id))
        for camera_id in sorted(found - listed):
            issues.append(Issue(
                "error", f"Images were found for '{camera_id}' but it has no "
                          f"entry in the camera file, so those images would "
                          f"have no location.", camera_id))
        return issues

    # ── Output ────────────────────────────────────────────────────────────────
    def normalised(self) -> pd.DataFrame:
        """The entries, cleaned into the exact shape ingestion expects."""
        frame = self.frame.copy()
        frame["camera_id"] = frame["camera_id"].astype(str).str.strip()
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame["habitat_type"] = (frame["habitat_type"].astype(str)
                                  .str.strip().str.lower()
                                  .where(lambda s: s.isin(HABITAT_CLASSES),
                                         "unknown"))
        return frame[COLUMNS]

    def write(self, path: str | Path) -> Path:
        """Write the camera file in the format ingestion reads."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.normalised().to_csv(path, index=False)
        return path

    def to_csv_bytes(self) -> bytes:
        """The file contents, for a download button."""
        return self.normalised().to_csv(index=False).encode()

    def lookup(self) -> dict[str, dict]:
        """camera_id -> its details, for ingestion to join against."""
        return {row["camera_id"]: row.to_dict()
                for _, row in self.normalised().iterrows()
                if row["camera_id"]}

    @property
    def camera_ids(self) -> list[str]:
        if self.frame.empty or "camera_id" not in self.frame.columns:
            return []
        return [c for c in self.frame["camera_id"].astype(str).str.strip() if c]


def blocking_issues(issues: list[Issue]) -> list[Issue]:
    return [issue for issue in issues if issue.blocking]


def warnings(issues: list[Issue]) -> list[Issue]:
    return [issue for issue in issues if not issue.blocking]
