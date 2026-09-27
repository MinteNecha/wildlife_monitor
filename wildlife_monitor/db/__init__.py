"""
Persistence package — the SQLite database in third normal form.

The database is the system of record. Pipelines write detections here,
Pipeline 2 reads its training sequences from here, and the dashboard reads
everything it displays from here. Large binaries stay on disk with their paths
stored in the tables.
"""

from wildlife_monitor.db.connection import (
    DB_PATH, connect, session, init_db, database_exists, table_counts,
)
from wildlife_monitor.db.repository import (
    Database, DetectionRepository, SpeciesRepository, CameraRepository,
    ImageRepository, PipelineRepository, SequenceRepository,
    BehaviourPatternRepository, ValidationRepository,
    load_detections, save_detections, save_patterns, load_images,
)

__all__ = [
    "DB_PATH", "connect", "session", "init_db", "database_exists",
    "table_counts", "Database", "DetectionRepository", "SpeciesRepository",
    "CameraRepository", "ImageRepository", "PipelineRepository",
    "SequenceRepository", "BehaviourPatternRepository", "ValidationRepository",
    "load_detections", "save_detections", "save_patterns", "load_images",
]
