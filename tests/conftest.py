"""
Shared fixtures.

Every test that touches the database gets its own temporary file, so the suite
never reads or writes the real data/wildlife.db.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wildlife_monitor.db.connection import connect, init_db


@pytest.fixture
def db_path(tmp_path):
    """A freshly created, empty database."""
    path = tmp_path / "test.db"
    init_db(path)
    return path


@pytest.fixture
def connection(db_path):
    """An open connection to the temporary database."""
    conn = connect(db_path)
    yield conn
    conn.close()


@pytest.fixture
def detections() -> pd.DataFrame:
    """A small, deterministic detections frame with known structure.

    Three cameras by design:
      HIGH  120 detections spread across the year   -> high site fidelity
      PEAK   20 detections all in March             -> temporally concentrated
      THIN    4 detections scattered                -> neither
    """
    rows = []
    rng = np.random.default_rng(0)

    def add(camera, count, months, hour):
        for index in range(count):
            month = int(months[index % len(months)])
            rows.append({
                "detection_id": f"{camera}-{index}",
                "image_id": f"{camera}_img{index}",
                "pipeline": "bioclip_megadetector",
                "timestamp": f"2010-{month:02d}-{index % 27 + 1:02d} "
                              f"{hour:02d}:30:00",
                "camera_id": camera,
                "latitude": -2.3, "longitude": 34.8,
                "habitat_type": "open_grassland",
                "species": "zebra",
                "confidence": 0.9,
                "location_type": "bbox", "location": "1,2,3,4",
                "detection_quality": 0.8,
                "instance_count": int(rng.integers(1, 4)),
                "image_path": f"images/{camera}_{index}.jpg",
                "mask_path": "", "overlay_path": "",
                "ground_truth_species": "zebra", "correct": "correct",
            })

    add("HIGH", 120, list(range(1, 13)), hour=9)
    add("PEAK", 20, [3], hour=22)
    add("THIN", 4, [5, 9], hour=7)
    return pd.DataFrame(rows)
