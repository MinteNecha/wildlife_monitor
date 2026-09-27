"""
Repositories over the normalised schema (Package P1).

Everything the system stores goes through here. A repository owns one table
(or one closely-bound pair) and exposes intention-revealing operations rather
than SQL, so callers never assemble statements themselves and the schema can
change in one place.

The pipelines emit a flat :class:`DetectionRecord`; saving one decomposes it
across Species, Camera, Image and Detection, creating the referenced rows if
they do not exist yet. Reading goes the other way through the ``DetectionFlat``
view, so consumers that expect a wide row — Pipeline 2, the dashboard — get
exactly the columns the old CSV had.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from wildlife_monitor.config import PROMPT_TEMPLATE
from wildlife_monitor.db.connection import connect, session

# Pipeline name -> the models behind it, for the Pipeline table.
PIPELINE_MODELS = {
    "bioclip_sam": ("BioCLIP", "SAM"),
    "bioclip_yolo": ("BioCLIP", "YOLOv11"),
    "bioclip_megadetector": ("BioCLIP", "MegaDetector"),
}

HABITAT_DESCRIPTIONS = {
    "open_grassland": "Open short-grass plains",
    "woodland": "Acacia and broadleaf woodland",
    "riverine": "Riparian corridor along a watercourse",
    "kopje": "Rocky granite outcrop",
    "unknown": "Habitat not recorded for this site",
}


def _as_dict(record: Any) -> dict[str, Any]:
    return asdict(record) if is_dataclass(record) else dict(record)


def _derive_season(image_id: str) -> str | None:
    """Season code from a Snapshot Serengeti image identifier, when present."""
    text = str(image_id).strip().upper()
    for prefix in ("S1", "S2", "S3", "S4"):
        if text.startswith(prefix + "/") or text.startswith(prefix + "_"):
            return f"S{prefix[1:].zfill(2)}"
    return None


class Repository:
    """Base class holding a connection shared by the concrete repositories."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def _scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        row = self.connection.execute(sql, params).fetchone()
        return row[0] if row else None


class SpeciesRepository(Repository):
    """The Species table — canonical labels and their BioCLIP prompts."""

    def ensure(self, canonical_name: str) -> int:
        """Return the id for a species, inserting it when first seen."""
        name = str(canonical_name).strip().lower()
        existing = self._scalar(
            "SELECT species_id FROM Species WHERE canonical_name = ?", (name,))
        if existing is not None:
            return int(existing)
        cursor = self.connection.execute(
            "INSERT INTO Species (canonical_name, common_name, text_prompt) "
            "VALUES (?, ?, ?)",
            (name, name.replace("_", " ").title(),
             PROMPT_TEMPLATE.format(species=name)))
        return int(cursor.lastrowid)

    def id_for(self, canonical_name: str) -> int | None:
        return self._scalar("SELECT species_id FROM Species "
                             "WHERE canonical_name = ?",
                             (str(canonical_name).strip().lower(),))

    def all_names(self) -> list[str]:
        return [row["canonical_name"] for row in self.connection.execute(
            "SELECT canonical_name FROM Species ORDER BY canonical_name")]


class CameraRepository(Repository):
    """The Camera and Habitat tables — site location and habitat context."""

    def ensure_habitat(self, habitat_type: str) -> str:
        habitat = (str(habitat_type).strip().lower() or "unknown")
        self.connection.execute(
            "INSERT OR IGNORE INTO Habitat (habitat_type, description) "
            "VALUES (?, ?)",
            (habitat, HABITAT_DESCRIPTIONS.get(habitat, "")))
        return habitat

    def ensure(self, camera_id: str, latitude: float = 0.0,
               longitude: float = 0.0, habitat_type: str = "unknown",
               season: str | None = None) -> str:
        """Insert a camera if unknown, filling gaps in an existing row."""
        camera = str(camera_id).strip()
        habitat = self.ensure_habitat(habitat_type)
        existing = self.connection.execute(
            "SELECT camera_id, season FROM Camera WHERE camera_id = ?",
            (camera,)).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO Camera (camera_id, site_name, latitude, longitude,"
                " habitat_type, season) VALUES (?, ?, ?, ?, ?, ?)",
                (camera, camera, float(latitude), float(longitude),
                 habitat, season))
        elif season and not existing["season"]:
            self.connection.execute(
                "UPDATE Camera SET season = ? WHERE camera_id = ?",
                (season, camera))
        return camera

    def frame(self) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT c.*, h.description AS habitat_description "
            "FROM Camera c LEFT JOIN Habitat h "
            "ON h.habitat_type = c.habitat_type", self.connection)


class ImageRepository(Repository):
    """The Image table — one row per photograph, with its ground truth."""

    def ensure(self, image_id: str, camera_id: str, captured_at: str,
               file_path: str, ground_truth_id: int | None = None,
               width: int | None = None, height: int | None = None,
               prepared: bool = False, upscaled: bool = False) -> str:
        image = str(image_id)
        existing = self.connection.execute(
            "SELECT image_id, ground_truth_id FROM Image WHERE image_id = ?",
            (image,)).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO Image (image_id, camera_id, captured_at, "
                "file_path, ground_truth_id, width, height, prepared, "
                "upscaled) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (image, str(camera_id), str(captured_at), str(file_path),
                 ground_truth_id, width, height,
                 int(bool(prepared)), int(bool(upscaled))))
        elif ground_truth_id is not None and existing["ground_truth_id"] is None:
            self.connection.execute(
                "UPDATE Image SET ground_truth_id = ? WHERE image_id = ?",
                (ground_truth_id, image))
        return image


class _ImageQueries(Repository):
    """Reading images back out, for steps that run over ingested photographs."""

    def frame(self, camera_id: str | None = None,
              unclassified_by: str | None = None) -> pd.DataFrame:
        """Ingested images joined to their camera.

        ``unclassified_by`` restricts the result to images that the named
        pipeline has not yet produced a detection for, so a classification run
        can be resumed without redoing work.
        """
        sql = ("SELECT i.image_id, i.camera_id, i.captured_at, i.file_path, "
               " i.width, i.height, COALESCE(i.upscaled, 0) AS upscaled, "
               " c.latitude, c.longitude, "
               " COALESCE(c.habitat_type, 'unknown') AS habitat_type "
               "FROM Image i JOIN Camera c ON c.camera_id = i.camera_id")
        clauses, params = [], []
        if camera_id:
            clauses.append("i.camera_id = ?")
            params.append(camera_id)
        if unclassified_by:
            clauses.append(
                "i.image_id NOT IN (SELECT d.image_id FROM Detection d "
                " JOIN Pipeline p ON p.pipeline_id = d.pipeline_id "
                " WHERE p.name = ?)")
            params.append(unclassified_by)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return pd.read_sql_query(sql + " ORDER BY i.camera_id, i.captured_at",
                                  self.connection, params=params)


class PipelineRepository(Repository):
    """The Pipeline table — which models produced a detection."""

    def ensure(self, name: str, version: str = "1.0") -> int:
        pipeline = str(name).strip()
        existing = self._scalar(
            "SELECT pipeline_id FROM Pipeline WHERE name = ?", (pipeline,))
        if existing is not None:
            return int(existing)
        retrieval, localiser = PIPELINE_MODELS.get(pipeline, ("", ""))
        cursor = self.connection.execute(
            "INSERT INTO Pipeline (name, retrieval_model, localiser_model, "
            "version) VALUES (?, ?, ?, ?)",
            (pipeline, retrieval, localiser, version))
        return int(cursor.lastrowid)


class DetectionRepository(Repository):
    """The Detection table — the system of record for pipeline output."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        super().__init__(connection)
        self.species = SpeciesRepository(connection)
        self.cameras = CameraRepository(connection)
        self.images = ImageRepository(connection)
        self.pipelines = PipelineRepository(connection)

    def save_record(self, record: Any) -> int:
        """Store one detection, creating the rows it references."""
        row = _as_dict(record)

        species_id = self.species.ensure(row["species"])
        ground_truth = str(row.get("ground_truth_species", "unknown")).lower()
        ground_truth_id = (None if ground_truth in ("", "unknown")
                           else self.species.ensure(ground_truth))

        camera_id = self.cameras.ensure(
            row.get("camera_id", ""), row.get("latitude", 0.0),
            row.get("longitude", 0.0), row.get("habitat_type", "unknown"),
            _derive_season(row.get("image_id", "")))
        self.images.ensure(row["image_id"], camera_id,
                            row.get("timestamp", ""), row.get("image_path", ""),
                            ground_truth_id)
        pipeline_id = self.pipelines.ensure(row.get("pipeline", "unknown"))

        confidence = float(row.get("confidence", 0.0) or 0.0)
        confidence = min(max(confidence, 0.0), 1.0)

        cursor = self.connection.execute(
            "INSERT INTO Detection (image_id, pipeline_id, species_id, "
            " confidence, detection_quality, location_type, location, "
            " instance_count, mask_path, overlay_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(image_id, pipeline_id) DO UPDATE SET "
            " species_id=excluded.species_id, confidence=excluded.confidence, "
            " detection_quality=excluded.detection_quality, "
            " location_type=excluded.location_type, location=excluded.location, "
            " instance_count=excluded.instance_count, "
            " mask_path=excluded.mask_path, overlay_path=excluded.overlay_path",
            (row["image_id"], pipeline_id, species_id, confidence,
             float(row.get("detection_quality", 0.0) or 0.0),
             row.get("location_type", ""), row.get("location", ""),
             int(row.get("instance_count", 1) or 1),
             row.get("mask_path", ""), row.get("overlay_path", "")))

        if cursor.lastrowid:
            return int(cursor.lastrowid)
        return int(self._scalar(
            "SELECT detection_id FROM Detection WHERE image_id = ? "
            "AND pipeline_id = ?", (row["image_id"], pipeline_id)))

    def save_many(self, records: Iterable[Any]) -> int:
        """Store a batch of detections, returning how many were written."""
        return sum(1 for record in records if self.save_record(record))

    # ── Reading ───────────────────────────────────────────────────────────────
    def frame(self, species: str | None = None,
              pipeline: str | None = None) -> pd.DataFrame:
        """Detections as the wide rows Pipeline 2 and the dashboard expect."""
        clauses, params = [], []
        if species:
            clauses.append("species = ?")
            params.append(str(species).strip().lower())
        if pipeline:
            clauses.append("pipeline = ?")
            params.append(pipeline)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return pd.read_sql_query(
            f"SELECT * FROM DetectionFlat{where} ORDER BY camera_id, timestamp",
            self.connection, params=params)

    def species_with_detections(self, pipeline: str | None = None) -> list[str]:
        sql = ("SELECT DISTINCT s.canonical_name AS name FROM Detection d "
               "JOIN Species s ON s.species_id = d.species_id")
        params: list[Any] = []
        if pipeline:
            sql += (" JOIN Pipeline p ON p.pipeline_id = d.pipeline_id "
                    "WHERE p.name = ?")
            params.append(pipeline)
        sql += " ORDER BY name"
        return [row["name"] for row in self.connection.execute(sql, params)]

    def pipelines_for(self, species: str) -> list[str]:
        return [row["name"] for row in self.connection.execute(
            "SELECT DISTINCT p.name AS name FROM Detection d "
            "JOIN Pipeline p ON p.pipeline_id = d.pipeline_id "
            "JOIN Species s ON s.species_id = d.species_id "
            "WHERE s.canonical_name = ? ORDER BY name",
            (str(species).lower(),))]


class SequenceRepository(Repository):
    """The Sequence table — one camera's detections for one species."""

    def ensure(self, species: str, camera_id: str, start_time: str = "",
               end_time: str = "", detection_count: int = 0,
               peak_month: int | None = None) -> int:
        species_id = SpeciesRepository(self.connection).ensure(species)
        # A sequence references a camera, so the camera row has to exist. It
        # normally does — detections created it — but a pattern can legitimately
        # arrive first, so create a minimal row rather than failing the insert.
        CameraRepository(self.connection).ensure(camera_id)
        existing = self._scalar(
            "SELECT sequence_id FROM Sequence WHERE species_id = ? "
            "AND camera_id = ?", (species_id, camera_id))
        if existing is not None:
            self.connection.execute(
                "UPDATE Sequence SET start_time = ?, end_time = ?, "
                "detection_count = ?, peak_month = ? WHERE sequence_id = ?",
                (start_time, end_time, detection_count, peak_month, existing))
            return int(existing)
        cursor = self.connection.execute(
            "INSERT INTO Sequence (species_id, camera_id, start_time, "
            "end_time, detection_count, peak_month) VALUES (?, ?, ?, ?, ?, ?)",
            (species_id, camera_id, start_time, end_time, detection_count,
             peak_month))
        return int(cursor.lastrowid)


class BehaviourPatternRepository(Repository):
    """The BehaviourPattern table — Pipeline 2's stored classifications."""

    def save(self, pattern: Any, model_version: str = "") -> int:
        row = _as_dict(pattern)
        sequences = SequenceRepository(self.connection)
        sequence_id = sequences.ensure(
            row.get("species", ""), row.get("camera_id", ""),
            start_time=str(row.get("start_time", "") or ""),
            end_time=str(row.get("end_time", "") or ""),
            detection_count=int(row.get("detection_count", 0) or 0),
            peak_month=row.get("peak_month"))
        version = model_version or row.get("model_version", "")

        self.connection.execute(
            "INSERT INTO BehaviourPattern (sequence_id, activity_class, "
            " movement_class, social_class, activity_confidence, "
            " movement_confidence, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(sequence_id, model_version) DO UPDATE SET "
            " activity_class=excluded.activity_class, "
            " movement_class=excluded.movement_class, "
            " social_class=excluded.social_class, "
            " activity_confidence=excluded.activity_confidence, "
            " movement_confidence=excluded.movement_confidence",
            (sequence_id, row.get("activity_class"), row.get("movement_class"),
             row.get("social_class"), row.get("activity_confidence"),
             row.get("movement_confidence"), version))
        return int(self._scalar(
            "SELECT pattern_id FROM BehaviourPattern WHERE sequence_id = ? "
            "AND model_version = ?", (sequence_id, version)))

    def save_many(self, patterns: Iterable[Any], model_version: str = "") -> int:
        return sum(1 for pattern in patterns
                   if self.save(pattern, model_version))

    def frame(self, species: str | None = None,
              model_version: str | None = None) -> pd.DataFrame:
        """Stored patterns joined back to their camera and species."""
        clauses, params = [], []
        if species:
            clauses.append("s.canonical_name = ?")
            params.append(str(species).lower())
        if model_version:
            clauses.append("b.model_version = ?")
            params.append(model_version)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return pd.read_sql_query(
            "SELECT b.pattern_id, q.camera_id, s.canonical_name AS species, "
            " b.activity_class, b.movement_class, b.social_class, "
            " b.activity_confidence, b.movement_confidence, b.model_version, "
            " q.detection_count, q.peak_month, c.habitat_type, "
            " c.latitude, c.longitude "
            "FROM BehaviourPattern b "
            "JOIN Sequence q ON q.sequence_id = b.sequence_id "
            "JOIN Species  s ON s.species_id  = q.species_id "
            "JOIN Camera   c ON c.camera_id   = q.camera_id"
            + where + " ORDER BY q.detection_count DESC",
            self.connection, params=params)


class ValidationRepository(Repository):
    """The Validation table — an ecologist's verdict on a pattern (FR7)."""

    VERDICTS = ("validated", "novel", "spurious")

    def record(self, pattern_id: int, verdict: str, notes: str = "") -> int:
        if verdict not in self.VERDICTS:
            raise ValueError(f"verdict must be one of {self.VERDICTS}")
        cursor = self.connection.execute(
            "INSERT INTO Validation (pattern_id, verdict, notes) "
            "VALUES (?, ?, ?)", (int(pattern_id), verdict, notes))
        return int(cursor.lastrowid)

    def for_pattern(self, pattern_id: int) -> pd.DataFrame:
        return pd.read_sql_query(
            "SELECT * FROM Validation WHERE pattern_id = ? "
            "ORDER BY validated_at DESC", self.connection, params=[pattern_id])

    def summary(self, species: str | None = None) -> pd.DataFrame:
        sql = ("SELECT v.verdict, COUNT(*) AS n FROM Validation v "
               "JOIN BehaviourPattern b ON b.pattern_id = v.pattern_id "
               "JOIN Sequence q ON q.sequence_id = b.sequence_id "
               "JOIN Species s ON s.species_id = q.species_id")
        params: list[Any] = []
        if species:
            sql += " WHERE s.canonical_name = ?"
            params.append(str(species).lower())
        sql += " GROUP BY v.verdict"
        return pd.read_sql_query(sql, self.connection, params=params)

    def latest_verdicts(self, species: str | None = None) -> pd.DataFrame:
        """Most recent verdict per pattern, for display alongside predictions."""
        sql = ("SELECT v.pattern_id, v.verdict, v.notes, v.validated_at, "
               " q.camera_id, s.canonical_name AS species "
               "FROM Validation v "
               "JOIN BehaviourPattern b ON b.pattern_id = v.pattern_id "
               "JOIN Sequence q ON q.sequence_id = b.sequence_id "
               "JOIN Species s ON s.species_id = q.species_id "
               "WHERE v.validation_id IN "
               " (SELECT MAX(validation_id) FROM Validation GROUP BY pattern_id)")
        params: list[Any] = []
        if species:
            sql += " AND s.canonical_name = ?"
            params.append(str(species).lower())
        return pd.read_sql_query(sql + " ORDER BY v.validated_at DESC",
                                  self.connection, params=params)


class Database:
    """Facade holding one connection and every repository over it."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.connection = connect(path)
        self.species = SpeciesRepository(self.connection)
        self.cameras = CameraRepository(self.connection)
        self.images = ImageRepository(self.connection)
        self.image_queries = _ImageQueries(self.connection)
        self.pipelines = PipelineRepository(self.connection)
        self.detections = DetectionRepository(self.connection)
        self.sequences = SequenceRepository(self.connection)
        self.patterns = BehaviourPatternRepository(self.connection)
        self.validations = ValidationRepository(self.connection)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.connection.rollback()
        self.close()


# ── One-shot helpers for callers that just want a frame ──────────────────────

def load_detections(species: str | None = None,
                    pipeline: str | None = None,
                    path: str | Path | None = None) -> pd.DataFrame:
    """Read detections without managing a connection."""
    with session(path) as connection:
        return DetectionRepository(connection).frame(species, pipeline)


def save_detections(records: Sequence[Any],
                    path: str | Path | None = None) -> int:
    """Write a batch of detection records in one transaction."""
    with session(path) as connection:
        return DetectionRepository(connection).save_many(records)


def load_images(camera_id: str | None = None,
                unclassified_by: str | None = None,
                path: str | Path | None = None) -> pd.DataFrame:
    """Ingested images, ready for a classification run."""
    with session(path) as connection:
        return _ImageQueries(connection).frame(camera_id, unclassified_by)


def save_patterns(patterns: Sequence[Any], model_version: str = "",
                  path: str | Path | None = None) -> int:
    """Persist Pipeline 2 classifications."""
    with session(path) as connection:
        return BehaviourPatternRepository(connection).save_many(
            patterns, model_version)
