"""
The path an external user takes: prepare, register cameras, ingest, classify.

Two behaviours are pinned down here above all others.

Enlarging an image must never be reported as a lossless fix. It lets a photo
pass the resolution check without adding any detail, and if that flag were
lost the resulting detections would be indistinguishable from good ones.

The animal detector must gate the species classifier. BioCLIP scores against a
closed candidate list and returns the nearest match for anything — including
an empty frame — so without the gate a photograph of grass becomes a confident
species record.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from wildlife_monitor.data.cameras import (
    CameraRegistry, blocking_issues, warnings,
)
from wildlife_monitor.data.ingestion import (
    ImageIngestor, discover_cameras, image_identifier,
)
from wildlife_monitor.data.preparation import ImagePreparer, describe_options
from wildlife_monitor.db import init_db, table_counts
from wildlife_monitor.pipelines.classify import SpeciesClassifier


def image_bytes(width: int, height: int, image_format: str = "JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (80, 110, 60)).save(buffer, image_format)
    return buffer.getvalue()


class Upload(io.BytesIO):
    """Stands in for a Streamlit upload, which has a name and is seekable."""

    def __init__(self, data: bytes, name: str) -> None:
        super().__init__(data)
        self.name = name


@pytest.fixture
def photo_tree(tmp_path) -> Path:
    """One folder per camera, with a mix of good and problem images."""
    root = tmp_path / "photos"
    layout = {
        "SiteA": [(1024, 768, "JPEG"), (1024, 768, "JPEG"), (800, 600, "JPEG")],
        "SiteB": [(1024, 768, "PNG"), (320, 240, "JPEG")],
    }
    for camera, specs in layout.items():
        (root / camera).mkdir(parents=True)
        for index, (width, height, fmt) in enumerate(specs):
            suffix = "png" if fmt == "PNG" else "jpg"
            (root / camera / f"IMG_{index:03}.{suffix}").write_bytes(
                image_bytes(width, height, fmt))
    return root


@pytest.fixture
def registry() -> CameraRegistry:
    return CameraRegistry.from_entries([
        {"camera_id": "SiteA", "latitude": -2.15, "longitude": 34.80,
         "habitat_type": "woodland"},
        {"camera_id": "SiteB", "latitude": -2.21, "longitude": 34.85,
         "habitat_type": "kopje"},
    ])


# ── Image preparation ────────────────────────────────────────────────────────

def test_format_conversion_is_lossless():
    result = ImagePreparer().prepare(io.BytesIO(image_bytes(1024, 768, "PNG")))
    assert result.prepared
    assert result.changed
    assert not result.cosmetic_only
    assert "JPEG" in result.summary


def test_oversized_images_are_shrunk_losslessly():
    result = ImagePreparer(max_dimension=2048).prepare(
        io.BytesIO(image_bytes(4000, 3000)))
    assert max(result.width, result.height) == 2048
    assert not result.cosmetic_only


def test_upscaling_is_refused_unless_explicitly_enabled():
    result = ImagePreparer().prepare(io.BytesIO(image_bytes(320, 240)))
    assert result.width == 320, "must not silently enlarge"
    assert not result.cosmetic_only


def test_upscaling_is_flagged_as_cosmetic_when_enabled():
    """The load-bearing test: an enlarged image must stay identifiable."""
    result = ImagePreparer(allow_upscale=True).prepare(
        io.BytesIO(image_bytes(320, 240)))
    assert (result.width, result.height) == (640, 480)
    assert result.cosmetic_only, "enlargement adds no detail and must say so"
    assert "no detail added" in result.summary
    assert result.to_details()["upscaled"] is True


def test_an_image_that_is_already_fine_is_left_alone():
    result = ImagePreparer(convert_format=False).prepare(
        io.BytesIO(image_bytes(1024, 768)))
    assert not result.changed
    assert result.summary == "No changes needed"


def test_unreadable_input_reports_rather_than_raises():
    result = ImagePreparer().prepare(io.BytesIO(b"not an image"))
    assert not result.prepared
    assert "Unreadable" in result.error


def test_recovery_options_do_not_oversell_enlargement():
    options = describe_options(640, 480)
    assert len(options) == 3
    enlargement = options[-1][1]
    assert "no better" in enlargement or "invents pixels" in enlargement


# ── Camera registry ──────────────────────────────────────────────────────────

def test_template_is_prefilled_from_discovered_folders():
    frame = CameraRegistry.template(["SiteA", "SiteB"])
    assert frame["camera_id"].tolist() == ["SiteA", "SiteB"]
    assert frame["latitude"].isna().all(), "coordinates are the user's to fill"


def test_valid_registry_has_no_issues(registry):
    assert registry.validate(["SiteA", "SiteB"]) == []


def test_blank_template_says_it_is_unfilled():
    empty = CameraRegistry(CameraRegistry.template(["SiteA"]))
    issues = blocking_issues(empty.validate())
    assert issues
    assert "no coordinates yet" in issues[0].message


@pytest.mark.parametrize("entry,fragment", [
    ({"camera_id": "A", "latitude": 200, "longitude": 34.8,
      "habitat_type": "woodland"}, "outside -90 to 90"),
    ({"camera_id": "A", "latitude": -2.1, "longitude": 400,
      "habitat_type": "woodland"}, "outside -180 to 180"),
    ({"camera_id": "A", "latitude": "x", "longitude": 34.8,
      "habitat_type": "woodland"}, "numeric"),
])
def test_bad_coordinates_are_rejected(entry, fragment):
    issues = blocking_issues(CameraRegistry.from_entries([entry]).validate())
    assert any(fragment in issue.message for issue in issues)


def test_duplicate_camera_names_are_rejected():
    duplicated = CameraRegistry.from_entries([
        {"camera_id": "A", "latitude": -2.1, "longitude": 34.8,
         "habitat_type": "woodland"},
        {"camera_id": "A", "latitude": -2.2, "longitude": 34.9,
         "habitat_type": "woodland"}])
    assert any("more than once" in issue.message
               for issue in blocking_issues(duplicated.validate()))


def test_images_without_a_camera_entry_block_ingestion(registry):
    """The quiet mistake: photos that would ingest with no location."""
    issues = blocking_issues(
        registry.validate(known_cameras=["SiteA", "SiteB", "SiteC"]))
    assert any("SiteC" in issue.message for issue in issues)


def test_a_camera_with_no_images_is_only_a_warning(registry):
    issues = registry.validate(known_cameras=["SiteA"])
    assert not blocking_issues(issues)
    assert any("SiteB" in issue.message for issue in warnings(issues))


def test_unknown_habitat_warns_and_falls_back():
    entry = CameraRegistry.from_entries([
        {"camera_id": "A", "latitude": -2.1, "longitude": 34.8,
         "habitat_type": "jungle"}])
    assert not blocking_issues(entry.validate())
    assert entry.normalised()["habitat_type"].iloc[0] == "unknown"


def test_alternative_column_names_are_understood(tmp_path):
    path = tmp_path / "mine.csv"
    pd.DataFrame({"Site": ["A1"], "Lat": [-2.1], "Lng": [34.8],
                  "Habitat": ["Woodland"]}).to_csv(path, index=False)
    normalised = CameraRegistry.load(path).normalised()
    assert normalised["camera_id"].iloc[0] == "A1"
    assert normalised["habitat_type"].iloc[0] == "woodland"


def test_registry_round_trips_through_a_file(registry, tmp_path):
    path = registry.write(tmp_path / "cameras.csv")
    assert CameraRegistry.load(path).camera_ids == ["SiteA", "SiteB"]


# ── Ingestion ────────────────────────────────────────────────────────────────

def test_cameras_are_discovered_from_folder_names(photo_tree):
    found = discover_cameras(photo_tree)
    assert set(found) == {"SiteA", "SiteB"}
    assert len(found["SiteA"]) == 3


def test_loose_images_are_grouped_rather_than_dropped(photo_tree):
    (photo_tree / "stray.jpg").write_bytes(image_bytes(1024, 768))
    found = discover_cameras(photo_tree)
    assert "unsorted" in found


def test_strict_ingestion_rejects_an_undersized_image(photo_tree, tmp_path,
                                                       registry):
    database = tmp_path / "strict.db"
    init_db(database)
    report = ImageIngestor(images_dir=tmp_path / "images",
                            db_path=database).ingest(photo_tree, registry)
    assert report.ingested == 4
    assert report.rejected_count == 1
    assert "Resolution" in report.rejected[0][1]


def test_preparation_rescues_what_it_honestly_can(photo_tree, tmp_path,
                                                   registry):
    database = tmp_path / "prepared.db"
    init_db(database)
    report = ImageIngestor(preparer=ImagePreparer(allow_upscale=True),
                            images_dir=tmp_path / "images",
                            db_path=database).ingest(photo_tree, registry)
    assert report.ingested == 5
    assert report.upscaled == 1, "only the undersized image was enlarged"


def test_the_upscale_flag_reaches_the_database(photo_tree, tmp_path, registry):
    import sqlite3

    database = tmp_path / "flagged.db"
    init_db(database)
    ImageIngestor(preparer=ImagePreparer(allow_upscale=True),
                  images_dir=tmp_path / "images",
                  db_path=database).ingest(photo_tree, registry)

    connection = sqlite3.connect(database)
    upscaled = connection.execute(
        "SELECT COUNT(*) FROM Image WHERE upscaled = 1").fetchone()[0]
    connection.close()
    assert upscaled == 1


def test_ingestion_is_idempotent(photo_tree, tmp_path, registry):
    database = tmp_path / "twice.db"
    init_db(database)
    ingestor = ImageIngestor(images_dir=tmp_path / "images", db_path=database)
    ingestor.ingest(photo_tree, registry)
    first = table_counts(database)["Image"]
    ingestor.ingest(photo_tree, registry)
    assert table_counts(database)["Image"] == first


def test_image_identifiers_are_stable_and_camera_scoped():
    assert image_identifier("SiteA", Path("IMG_001.jpg")) == "SiteA_IMG_001"
    assert (image_identifier("SiteA", Path("a.jpg"))
            != image_identifier("SiteB", Path("a.jpg")))


def test_browser_uploads_ingest_under_the_chosen_camera(tmp_path, registry):
    database = tmp_path / "uploads.db"
    init_db(database)
    files = [Upload(image_bytes(1024, 768), "a.jpg"),
             Upload(image_bytes(320, 240), "small.jpg")]
    report = ImageIngestor(images_dir=tmp_path / "images",
                            db_path=database).ingest_uploads(
        files, "SiteA", registry)
    assert report.ingested == 1
    assert report.rejected_count == 1


# ── Classification ───────────────────────────────────────────────────────────

class StubDetector:
    def __init__(self, count: int) -> None:
        self.count = count
        self.calls = 0

    def detect_all(self, image_rgb):
        self.calls += 1
        if self.count == 0:
            return [], 0
        return [(10, 10, 90, 90, 0.93)], self.count


class StubRecogniser:
    """Always returns a confident answer — exactly what the real model does."""

    def __init__(self) -> None:
        self.calls = 0

    def classify(self, image_path, candidates, top_k=5, text_features=None):
        self.calls += 1
        ordered = sorted(candidates)
        return [(ordered[0], 0.82)][:top_k]


@pytest.fixture
def image_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {"image_id": "A_1", "camera_id": "SiteA", "image_path": "/x/a.jpg",
         "captured_at": "2024-03-01 09:00:00", "latitude": -2.1,
         "longitude": 34.8, "habitat_type": "woodland"},
        {"image_id": "A_2", "camera_id": "SiteA", "image_path": "/x/b.jpg",
         "captured_at": "2024-03-02 21:00:00", "latitude": -2.1,
         "longitude": 34.8, "habitat_type": "woodland"},
    ])


def classifier(detector, recogniser, **kwargs) -> SpeciesClassifier:
    return SpeciesClassifier(detector=detector, recogniser=recogniser,
                              image_loader=lambda path: None
                              if "missing" in path else "IMAGE", **kwargs)


def test_empty_frames_never_reach_the_recogniser(image_frame):
    """The gate's whole purpose: grass must not become a species record."""
    detector, recogniser = StubDetector(0), StubRecogniser()
    records, report = classifier(detector, recogniser).classify_frame(image_frame)

    assert records == []
    assert report.empty_frames == 2
    assert recogniser.calls == 0, "BioCLIP must not be asked about empty frames"


def test_without_the_gate_empty_frames_are_confidently_labelled(image_frame):
    """Shows what the gate prevents, so the guard is not removed casually."""
    records, _ = classifier(StubDetector(0), StubRecogniser(),
                             use_gate=False).classify_frame(image_frame)
    assert len(records) == 2
    assert all(record.confidence > 0.5 for record in records)


def test_frames_with_animals_produce_records(image_frame):
    records, report = classifier(StubDetector(2),
                                  StubRecogniser()).classify_frame(image_frame)
    assert len(records) == 2
    assert report.with_animals == 2
    assert records[0].instance_count == 2
    assert records[0].location == "10,10,90,90"


def test_classified_records_are_unverified(image_frame):
    """An external user has no labels, so nothing may claim to be correct."""
    records, _ = classifier(StubDetector(1),
                             StubRecogniser()).classify_frame(image_frame)
    assert all(record.correct == "unknown" for record in records)
    assert all(record.ground_truth_species == "unknown" for record in records)


def test_unreadable_images_are_not_counted_as_empty(image_frame):
    broken = pd.DataFrame([{"image_id": "A_3", "camera_id": "SiteA",
                            "image_path": "/x/missing.jpg"}])
    _, report = classifier(StubDetector(1),
                            StubRecogniser()).classify_frame(broken)
    assert report.unreadable == 1
    assert report.empty_frames == 0


@pytest.mark.parametrize("candidates,fragment", [
    (None, "all 13"),
    (["zebra"], "zebra only"),
    (["zebra", "wildebeest", "buffalo"], "3 species"),
])
def test_the_three_modes_describe_themselves(candidates, fragment):
    built = SpeciesClassifier(candidates=candidates)
    assert fragment in built.mode


def test_a_shortlist_limits_what_can_be_predicted(image_frame):
    shortlist = ["zebra", "wildebeest"]
    records, _ = classifier(StubDetector(1), StubRecogniser(),
                             candidates=shortlist).classify_frame(image_frame)
    assert all(record.species in shortlist for record in records)
