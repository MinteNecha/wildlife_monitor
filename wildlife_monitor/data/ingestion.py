"""
Folder ingestion (Package P1, FR1).

Takes a folder of camera trap photographs and puts them into the system, with
no Snapshot Serengeti metadata required. The expected layout is one folder per
camera, which is how camera trap users already organise their files:

    photos/
      SiteA/  IMG_0001.JPG  IMG_0002.JPG  ...
      SiteB/  ...

The folder name becomes the camera identifier. Capture times come from EXIF,
which camera traps write reliably. Location and habitat come from the camera
registry. Nothing here needs to know what species is in any photograph — that
is what the classification step is for.

Ingestion is idempotent. Image identifiers are derived from the camera and the
file name, so re-running over the same folder updates rather than duplicates.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from wildlife_monitor.config import IMAGES_DIR
from wildlife_monitor.data.cameras import CameraRegistry
from wildlife_monitor.data.preparation import ImagePreparer, PreparationResult
from wildlife_monitor.data.validator import ImageValidator
from wildlife_monitor.db import Database

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


@dataclass
class IngestionReport:
    """What ingestion did, and what it could not do."""

    cameras: int = 0
    found: int = 0
    ingested: int = 0
    rejected: list[tuple[str, str]] = field(default_factory=list)
    adjusted: int = 0
    upscaled: int = 0
    undated: int = 0
    first_capture: str = ""
    last_capture: str = ""

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)

    def summary_line(self) -> str:
        parts = [f"{self.ingested:,} of {self.found:,} images ingested",
                 f"{self.cameras} camera{'' if self.cameras == 1 else 's'}"]
        if self.rejected_count:
            parts.append(f"{self.rejected_count:,} rejected")
        if self.adjusted:
            parts.append(f"{self.adjusted:,} adjusted")
        if self.upscaled:
            parts.append(f"{self.upscaled:,} enlarged (no detail added)")
        if self.undated:
            parts.append(f"{self.undated:,} without a capture time")
        return " · ".join(parts)


def discover_cameras(root: str | Path) -> dict[str, list[Path]]:
    """Map camera folder names to the images inside them.

    Images sitting loose in the root are grouped under 'unsorted' rather than
    being dropped, so a user who has not organised into folders still gets a
    result — with one camera, which the sufficiency check will flag.
    """
    root = Path(root)
    if not root.exists():
        return {}

    cameras: dict[str, list[Path]] = {}
    loose = [path for path in sorted(root.iterdir())
             if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    if loose:
        cameras["unsorted"] = loose

    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        images = [path for path in sorted(directory.rglob("*"))
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
        if images:
            cameras[directory.name] = images
    return cameras


def image_identifier(camera_id: str, path: Path) -> str:
    """A stable identifier, so re-ingesting the same file updates one row."""
    return f"{camera_id}_{path.stem}".replace(" ", "_")


class ImageIngestor:
    """Validates, prepares and records a folder of camera trap photographs."""

    def __init__(self, validator: ImageValidator | None = None,
                 preparer: ImagePreparer | None = None,
                 images_dir: Path | None = None,
                 db_path: str | Path | None = None) -> None:
        self.validator = validator or ImageValidator()
        self.preparer = preparer
        self.images_dir = Path(images_dir) if images_dir else IMAGES_DIR
        self.db_path = db_path

    def ingest(self, root: str | Path, registry: CameraRegistry,
               dry_run: bool = False) -> IngestionReport:
        """Ingest every camera folder under ``root``."""
        report = IngestionReport()
        cameras = discover_cameras(root)
        report.cameras = len(cameras)
        report.found = sum(len(paths) for paths in cameras.values())
        if not cameras:
            return report

        lookup = registry.lookup()
        captures: list[str] = []

        with Database(self.db_path) as database:
            for camera_id, paths in cameras.items():
                details = lookup.get(camera_id, {})
                if not dry_run:
                    database.cameras.ensure(
                        camera_id,
                        details.get("latitude") or 0.0,
                        details.get("longitude") or 0.0,
                        details.get("habitat_type") or "unknown")

                for path in paths:
                    captured = self._ingest_one(
                        database, camera_id, path, report, dry_run)
                    if captured:
                        captures.append(captured)

        if captures:
            captures.sort()
            report.first_capture, report.last_capture = captures[0], captures[-1]
        return report

    def _ingest_one(self, database, camera_id: str, path: Path,
                    report: IngestionReport, dry_run: bool) -> str:
        """Handle one photograph. Returns its capture time, if it has one."""
        result = self.validator.validate(path)
        prepared: PreparationResult | None = None

        if not result.valid and self.preparer is not None:
            prepared = self.preparer.prepare(path)
            if prepared.prepared and self.preparer.meets_resolution(
                    prepared.width, prepared.height):
                result = None       # preparation rescued it
            else:
                prepared = None

        if result is not None and not result.valid:
            report.rejected.append((path.name, result.reason))
            return ""

        if prepared is None and self.preparer is not None:
            candidate = self.preparer.prepare(path)
            if candidate.prepared and candidate.changed:
                prepared = candidate

        captured = self.validator.extract_timestamp(path)
        captured_at = captured.isoformat(sep=" ") if captured else ""
        if not captured_at:
            report.undated += 1

        if prepared is not None:
            report.adjusted += 1
            if prepared.cosmetic_only:
                report.upscaled += 1

        if dry_run:
            report.ingested += 1
            return captured_at

        destination = self._store(camera_id, path, prepared)
        width = prepared.width if prepared else (
            result.details.get("width") if result else None)
        height = prepared.height if prepared else (
            result.details.get("height") if result else None)

        database.images.ensure(
            image_identifier(camera_id, path), camera_id, captured_at,
            str(destination), None, width, height,
            prepared=prepared is not None,
            upscaled=bool(prepared and prepared.cosmetic_only))
        report.ingested += 1
        return captured_at

    def ingest_uploads(self, files, camera_id: str,
                       registry: CameraRegistry) -> IngestionReport:
        """Ingest browser uploads, which arrive as file objects, not paths.

        A browser upload carries no folder, so the camera is chosen on the
        page and applies to the whole batch. That is also the honest workflow
        for a person uploading by hand — one camera's card at a time. Bulk
        folder ingestion belongs on the command line.
        """
        report = IngestionReport(cameras=1, found=len(files))
        details = registry.lookup().get(camera_id, {})
        captures: list[str] = []

        with Database(self.db_path) as database:
            database.cameras.ensure(
                camera_id, details.get("latitude") or 0.0,
                details.get("longitude") or 0.0,
                details.get("habitat_type") or "unknown")

            for file in files:
                captured = self._ingest_upload(
                    database, camera_id, file, report)
                if captured:
                    captures.append(captured)

        if captures:
            captures.sort()
            report.first_capture, report.last_capture = captures[0], captures[-1]
        return report

    def _ingest_upload(self, database, camera_id: str, file,
                       report: IngestionReport) -> str:
        name = getattr(file, "name", "upload")
        stem = Path(name).stem

        result = self.validator.validate(file, name=name)
        prepared: PreparationResult | None = None

        if self.preparer is not None:
            candidate = self.preparer.prepare(file, name)
            rescued = (candidate.prepared
                       and self.preparer.meets_resolution(candidate.width,
                                                          candidate.height))
            if candidate.prepared and (candidate.changed or not result.valid):
                if result.valid or rescued:
                    prepared = candidate
                    result = None

        if result is not None and not result.valid:
            report.rejected.append((name, result.reason))
            return ""

        captured = self.validator.extract_timestamp(file)
        captured_at = captured.isoformat(sep=" ") if captured else ""
        if not captured_at:
            report.undated += 1

        folder = self.images_dir / camera_id
        folder.mkdir(parents=True, exist_ok=True)

        if prepared is not None and prepared.data is not None:
            report.adjusted += 1
            if prepared.cosmetic_only:
                report.upscaled += 1
            destination = folder / f"{stem}.jpg"
            destination.write_bytes(prepared.data)
            width, height = prepared.width, prepared.height
        else:
            destination = folder / name
            file.seek(0)
            destination.write_bytes(file.read())
            width = result.details.get("width") if result else None
            height = result.details.get("height") if result else None

        database.images.ensure(
            f"{camera_id}_{stem}".replace(" ", "_"), camera_id, captured_at,
            str(destination), None, width, height,
            prepared=prepared is not None,
            upscaled=bool(prepared and prepared.cosmetic_only))
        report.ingested += 1
        return captured_at

    def _store(self, camera_id: str, path: Path,
               prepared: PreparationResult | None) -> Path:
        """Copy the image into the project, per camera."""
        folder = self.images_dir / camera_id
        folder.mkdir(parents=True, exist_ok=True)
        if prepared is not None and prepared.data is not None:
            destination = folder / f"{path.stem}.jpg"
            destination.write_bytes(prepared.data)
        else:
            destination = folder / path.name
            if path.resolve() != destination.resolve():
                shutil.copy2(path, destination)
        return destination
