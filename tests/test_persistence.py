"""
Database, query parsing, validation, and the supporting P1/P4 classes.

The round-trip test is the important one: the pipelines now write to a
normalised schema, and reading back through the flat view has to reproduce
what the flat CSV used to hold, or every downstream consumer silently changes
meaning.
"""

from __future__ import annotations

import pandas as pd
import pytest

from wildlife_monitor.config.validator import ConfigValidator
from wildlife_monitor.data.export import ExportService
from wildlife_monitor.db.repository import (
    Database, DetectionRepository, ValidationRepository,
)
from wildlife_monitor.pipeline2.query import QueryEngine
from wildlife_monitor.pipeline2.validation import PatternValidator
from wildlife_monitor.utils.records import DetectionRecord


# ── Database ─────────────────────────────────────────────────────────────────

def test_schema_creates_every_table(connection):
    tables = {row["name"] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"Camera", "Habitat", "Species", "Image", "Pipeline", "Detection",
            "Prediction", "Sequence", "SequenceMember", "BehaviourPattern",
            "Validation"} <= tables


def test_detection_round_trip_preserves_the_flat_row(connection, detections):
    repository = DetectionRepository(connection)
    repository.save_many(detections.to_dict(orient="records"))
    connection.commit()

    stored = repository.frame("zebra", "bioclip_megadetector")
    assert len(stored) == len(detections)

    original = detections.sort_values("image_id").reset_index(drop=True)
    restored = stored.sort_values("image_id").reset_index(drop=True)

    # detection_id becomes a surrogate key, so it is excluded by design.
    for column in ["image_id", "camera_id", "species", "timestamp",
                   "location_type", "location", "instance_count",
                   "habitat_type", "correct", "ground_truth_species"]:
        assert (original[column].astype(str).tolist()
                == restored[column].astype(str).tolist()), f"{column} changed"


def test_reimporting_the_same_detections_does_not_duplicate(connection, detections):
    repository = DetectionRepository(connection)
    records = detections.to_dict(orient="records")
    repository.save_many(records)
    repository.save_many(records)
    connection.commit()
    assert len(repository.frame("zebra")) == len(detections)


def test_normalisation_stores_each_camera_once(connection, detections):
    DetectionRepository(connection).save_many(detections.to_dict(orient="records"))
    connection.commit()
    cameras = connection.execute("SELECT COUNT(*) AS n FROM Camera").fetchone()["n"]
    assert cameras == detections.camera_id.nunique() == 3


def test_a_detection_record_dataclass_can_be_stored(connection):
    record = DetectionRecord(
        detection_id="x1", image_id="img-1", pipeline="bioclip_yolo",
        timestamp="2010-04-01 08:00:00", camera_id="B04",
        latitude=-2.1, longitude=34.8, habitat_type="woodland",
        species="buffalo", confidence=0.77, location_type="bbox",
        location="1,2,3,4", detection_quality=0.5, instance_count=2,
        image_path="images/x.jpg", ground_truth_species="buffalo",
        correct="correct")
    DetectionRepository(connection).save_record(record)
    connection.commit()
    stored = DetectionRepository(connection).frame("buffalo")
    assert len(stored) == 1
    assert stored.iloc[0]["instance_count"] == 2


def test_confidence_outside_zero_to_one_is_clamped(connection, detections):
    row = detections.to_dict(orient="records")[0]
    row["confidence"] = 4.2
    DetectionRepository(connection).save_record(row)
    connection.commit()
    assert DetectionRepository(connection).frame().iloc[0]["confidence"] <= 1.0


def test_validation_rejects_an_unknown_verdict(connection):
    with pytest.raises(ValueError):
        ValidationRepository(connection).record(1, "excellent")


# ── Query engine (FR6) ───────────────────────────────────────────────────────

@pytest.fixture
def patterns() -> pd.DataFrame:
    return pd.DataFrame([
        {"pattern_id": 1, "camera_id": "B04", "species": "zebra",
         "activity_class": "nocturnal", "movement_class": "migratory",
         "social_class": "large herd", "movement_confidence": 0.91,
         "activity_confidence": 0.8, "detection_count": 80,
         "habitat_type": "woodland", "peak_month": 7},
        {"pattern_id": 2, "camera_id": "C05", "species": "zebra",
         "activity_class": "diurnal", "movement_class": "territorial",
         "social_class": "solitary", "movement_confidence": 0.55,
         "activity_confidence": 0.7, "detection_count": 12,
         "habitat_type": "kopje", "peak_month": 3},
        {"pattern_id": 3, "camera_id": "D06", "species": "buffalo",
         "activity_class": "diurnal", "movement_class": "nomadic",
         "social_class": "small group", "movement_confidence": 0.62,
         "activity_confidence": 0.6, "detection_count": 40,
         "habitat_type": "woodland", "peak_month": 7},
    ])


@pytest.mark.parametrize("query,expected_cameras", [
    ("nocturnal zebra", ["B04"]),
    ("migratory", ["B04"]),
    ("patterns in woodland", ["B04", "D06"]),
    ("large herds", ["B04"]),
    ("buffalo", ["D06"]),
    ("cameras with more than 50 detections", ["B04"]),
    ("confidence above 0.9", ["B04"]),
    ("peaking in July", ["B04", "D06"]),
    ("camera C05", ["C05"]),
])
def test_queries_select_the_right_patterns(patterns, query, expected_cameras):
    results, _ = QueryEngine().run(query, patterns)
    assert sorted(results["camera_id"].tolist()) == sorted(expected_cameras)


def test_query_reports_what_it_understood(patterns):
    _, parsed = QueryEngine().run("nocturnal zebra in woodland", patterns)
    assert parsed.activity == "nocturnal"
    assert parsed.species == "zebra"
    assert parsed.habitat == "woodland"
    assert "nocturnal" in parsed.describe()


def test_unrecognised_query_returns_everything_and_says_so(patterns):
    results, parsed = QueryEngine().run("wombats doing cartwheels", patterns)
    assert parsed.is_empty
    assert len(results) == len(patterns)
    assert "wombats" in parsed.unrecognised


def test_everyday_synonyms_are_understood(patterns):
    _, night = QueryEngine().run("animals active at night", patterns)
    assert night.activity == "nocturnal"
    _, resident = QueryEngine().run("resident cameras", patterns)
    assert resident.movement == "territorial"


def test_limit_caps_the_result_count(patterns):
    results, parsed = QueryEngine().run("top 2 zebra", patterns)
    assert parsed.limit == 2
    assert len(results) <= 2


# ── Pattern validation (FR7) ─────────────────────────────────────────────────

def _seed_pattern(db_path) -> int:
    with Database(db_path) as database:
        return database.patterns.save({
            "camera_id": "B04", "species": "zebra",
            "activity_class": "nocturnal", "movement_class": "migratory",
            "social_class": "large herd", "activity_confidence": 0.7,
            "movement_confidence": 0.6, "detection_count": 80,
            "peak_month": 7}, model_version="LSTM · zebra · test")


def test_validated_verdicts_raise_adjusted_confidence(db_path):
    pattern_id = _seed_pattern(db_path)
    validator = PatternValidator(db_path)
    assert validator.update_confidence(pattern_id) == pytest.approx(0.6)

    validator.record_verdict(pattern_id, "validated", "matches field notes")
    raised = validator.update_confidence(pattern_id)
    assert raised > 0.6


def test_spurious_verdicts_lower_adjusted_confidence(db_path):
    pattern_id = _seed_pattern(db_path)
    validator = PatternValidator(db_path)
    validator.record_verdict(pattern_id, "spurious", "camera was faulty")
    assert validator.update_confidence(pattern_id) < 0.6


def test_novel_verdict_records_interest_without_moving_confidence(db_path):
    pattern_id = _seed_pattern(db_path)
    validator = PatternValidator(db_path)
    validator.record_verdict(pattern_id, "novel", "not described before")
    assert validator.update_confidence(pattern_id) == pytest.approx(0.6)
    assert validator.summary(pattern_id).verdicts["novel"] == 1


def test_model_confidence_is_never_overwritten_by_a_verdict(db_path):
    pattern_id = _seed_pattern(db_path)
    validator = PatternValidator(db_path)
    validator.record_verdict(pattern_id, "spurious")
    summary = validator.summary(pattern_id)
    assert summary.model_confidence == pytest.approx(0.6)
    assert summary.adjusted_confidence != summary.model_confidence


def test_verdict_history_is_appended_not_replaced(db_path):
    pattern_id = _seed_pattern(db_path)
    validator = PatternValidator(db_path)
    validator.record_verdict(pattern_id, "validated")
    validator.record_verdict(pattern_id, "spurious", "changed my mind")
    summary = validator.summary(pattern_id)
    assert summary.review_count == 2
    assert summary.latest_verdict == "spurious"


def test_unknown_verdict_is_refused(db_path):
    pattern_id = _seed_pattern(db_path)
    with pytest.raises(ValueError, match="verdict must be"):
        PatternValidator(db_path).record_verdict(pattern_id, "brilliant")


# ── Configuration validation (FR9) ───────────────────────────────────────────

@pytest.mark.parametrize("key,value,valid", [
    ("threshold", 0.5, True), ("threshold", 1.5, False),
    ("threshold", -0.1, False),
    ("top_n", 15, True), ("top_n", 0, False),
    ("arch", "LSTM", True), ("arch", "GRU", False),
    ("device", "cpu", True), ("device", "quantum", False),
    ("movement", "migratory, territorial, nomadic", True),
    ("movement", "migratory", False),
    ("movement", "same, same", False),
])
def test_config_validation(key, value, valid):
    assert bool(ConfigValidator().validate(key, value)) is valid


def test_rejected_setting_states_the_valid_range():
    result = ConfigValidator().validate("threshold", 9.9)
    assert not result.valid
    assert "between 0.0 and 1.0" in result.reason
    assert ConfigValidator().valid_range("threshold") == (0.0, 1.0)


def test_unknown_settings_pass_through_unchecked():
    assert ConfigValidator().validate("something_new", object()).valid


# ── Export (FR8) ─────────────────────────────────────────────────────────────

def test_csv_and_json_exports_round_trip(detections):
    service = ExportService()
    frame = pd.read_csv(pd.io.common.BytesIO(service.to_csv(detections)))
    assert len(frame) == len(detections)

    import json
    payload = json.loads(service.to_json(detections.head(3)))
    assert payload["row_count"] == 3
    assert len(payload["records"]) == 3


def test_pdf_export_is_a_valid_document(detections, tmp_path):
    path = ExportService().export_pdf(
        detections, tmp_path / "report.pdf", title="Test Report",
        summary=[("Detections", len(detections))])
    content = path.read_bytes()
    assert content.startswith(b"%PDF-1.4")
    assert content.rstrip().endswith(b"%%EOF")
    assert b"/Type /Catalog" in content and b"xref" in content


def test_pdf_paginates_long_tables(detections, tmp_path):
    content = ExportService().to_pdf(detections, max_rows=200)
    assert content.count(b"/Type /Page\n") >= 2 or content.count(b"/Type /Page ") >= 2
