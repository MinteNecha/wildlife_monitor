"""
Capture-time resolution: each source, the order between them, and the limits.

Every behavioural output depends on when a photograph was taken, so the tests
that matter most here are the ones that pin down what the system refuses to
guess:

  * file modification time is never consulted, however convenient it looks,
    because copying a folder resets it
  * a date with no time of day is not silently filled in with midnight
  * a name that identifies two different capture times answers neither

and the one that makes the chain worth having: a curated file beats whatever
survived in the photograph itself.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from wildlife_monitor.data.cameras import CameraRegistry
from wildlife_monitor.data.ingestion import ImageIngestor
from wildlife_monitor.data.timestamps import (
    ANNOTATIONS, EXIF, FILENAME, FILENAME_DATE, METADATA, NONE,
    AnnotationFile, FilenameTimestamps, MetadataFile, TimestampResolver,
    build_resolver, parse_timestamp,
)
from wildlife_monitor.db import init_db

MOMENT = datetime(2010, 7, 20, 6, 14, 6)


# ── helpers ──────────────────────────────────────────────────────────────────

def upload(data: bytes, name: str) -> io.BytesIO:
    """Stands in for a Streamlit upload, which has a name and is seekable."""
    handle = io.BytesIO(data)
    handle.name = name                                  # type: ignore[attr-defined]
    return handle


def csv_upload(text: str, name: str = "times.csv") -> io.BytesIO:
    return upload(text.encode(), name)


def json_upload(document: dict, name: str = "annotations.json") -> io.BytesIO:
    return upload(json.dumps(document).encode(), name)


def serengeti_document() -> dict:
    """The shape of SnapshotSerengetiS01.json, in miniature.

    Both places the format may carry a capture time are represented: the
    ``images`` entry and the ``annotations`` entry that Snapshot Serengeti
    duplicates it onto, which is where ``extract_ground_truth.py`` reads it.
    """
    return {
        "categories": [{"id": 0, "name": "empty"},
                        {"id": 1, "name": "zebra"},
                        {"id": 2, "name": "wildebeest"}],
        "images": [
            {"id": "S1/B04/B04_R1/S1_B04_R1_PICT0001",
             "file_name": "S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG",
             "location": "B04", "datetime": "2010-07-20 06:14:06"},
            # No datetime here — it has to come from the annotation below.
            {"id": "S1/B04/B04_R1/S1_B04_R1_PICT0002",
             "file_name": "S1/B04/B04_R1/S1_B04_R1_PICT0002.JPG",
             "location": "B04"},
        ],
        "annotations": [
            {"image_id": "S1/B04/B04_R1/S1_B04_R1_PICT0001",
             "category_id": 1, "count": "2",
             "datetime": "2010-07-20 06:14:06"},
            {"image_id": "S1/B04/B04_R1/S1_B04_R1_PICT0002",
             "category_id": 0, "datetime": "2010-07-21 18:02:11"},
        ],
    }


# ── text parsing ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "2010-07-20 06:14:06",
    "2010-07-20T06:14:06",
    "2010:07:20 06:14:06",           # the EXIF spelling
    "2010/07/20 06:14:06",
    "20/07/2010 06:14:06",           # day-first
    "20100720 061406",
    "2010-07-20T06:14:06Z",
    "2010-07-20T06:14:06.500+02:00",  # fractional seconds and an offset
])
def test_parse_timestamp_accepts_the_formats_tools_export(text):
    assert parse_timestamp(text) == MOMENT


@pytest.mark.parametrize("text", ["", None, "nan", "NULL", "not a date",
                                   "2010-13-45 99:99:99", "yesterday"])
def test_parse_timestamp_returns_none_rather_than_raising(text):
    assert parse_timestamp(text) is None


def test_slash_dates_prefer_day_first_but_fall_back():
    """07/08 is ambiguous; 20/07 is not, and 12/25 can only be month-first."""
    assert parse_timestamp("20/07/2010 06:14:06") == MOMENT
    assert parse_timestamp("07/08/2010 06:14:06").month == 8      # day-first
    assert parse_timestamp("12/25/2010 06:14:06").month == 12     # month-first


# ── the annotation file ──────────────────────────────────────────────────────

def test_annotation_file_reads_times_from_images_and_annotations():
    annotations = AnnotationFile(serengeti_document())
    assert annotations.report.entries == 2
    # First image: the time is on the images entry.
    assert annotations.lookup(
        "S1/B04/B04_R1/S1_B04_R1_PICT0001.JPG") == MOMENT
    # Second: only the annotations entry has it.
    assert annotations.lookup(
        "S1_B04_R1_PICT0002.JPG") == datetime(2010, 7, 21, 18, 2, 11)


def test_annotation_file_matches_the_flattened_name_on_disk():
    """download_subset_images.py flattens the nested path; both must match."""
    annotations = AnnotationFile(serengeti_document())
    flattened = "s1_b04_b04_r1_s1_b04_r1_pict0001.jpg"
    assert annotations.lookup(flattened) == MOMENT


def test_annotation_file_supplies_species_labels_but_not_for_empty_frames():
    annotations = AnnotationFile(serengeti_document())
    assert annotations.has_labels
    assert annotations.label_for("S1_B04_R1_PICT0001.JPG") == "zebra"
    # category 'empty' is not a species and must not become ground truth.
    assert annotations.label_for("S1_B04_R1_PICT0002.JPG") is None


def test_a_bare_frame_number_resolves_only_when_the_camera_matches():
    """PICT0001 repeats across sites, so it is a camera-scoped key only."""
    annotations = AnnotationFile(serengeti_document())
    assert annotations.lookup("PICT0001.JPG") is None
    assert annotations.lookup("PICT0001.JPG", camera_id="B04") == MOMENT


def test_annotation_file_reports_a_file_it_cannot_use_instead_of_raising():
    broken = AnnotationFile.load(upload(b"{ not json", "bad.json"))
    assert not broken.report.usable
    assert "not valid JSON" in broken.report.problems[0]

    empty = AnnotationFile.load(json_upload({"images": [], "annotations": []}))
    assert not empty.report.usable
    assert "datetime" in empty.report.problems[0]

    missing = AnnotationFile.load(Path("no-such-file.json"))
    assert missing.report.problems == ["file not found"]


# ── the metadata CSV ─────────────────────────────────────────────────────────

def test_metadata_csv_accepts_the_column_names_tools_actually_export():
    for header in ["filename,timestamp", "file_name,datetime",
                   "Image,DateTimeOriginal", "image_id,real_datetime"]:
        metadata = MetadataFile.load(
            csv_upload(f"{header}\nIMG_0001.JPG,2010-07-20 06:14:06\n"))
        assert metadata.lookup("IMG_0001.JPG") == MOMENT, header


def test_metadata_csv_says_what_is_missing_rather_than_failing_silently():
    metadata = MetadataFile.load(csv_upload("a,b\n1,2\n"))
    assert not metadata.report.usable
    assert "file name column" in metadata.report.problems[0]
    assert "timestamp column" in metadata.report.problems[0]


def test_metadata_csv_counts_rows_it_could_not_read():
    metadata = MetadataFile.load(csv_upload(
        "filename,timestamp\n"
        "a.jpg,2010-07-20 06:14:06\n"
        "b.jpg,sometime last winter\n"))
    assert metadata.report.entries == 1
    assert "1 row had a timestamp that could not be read" in \
        metadata.report.problems


def test_a_camera_column_separates_file_names_that_repeat():
    metadata = MetadataFile.load(csv_upload(
        "filename,timestamp,camera\n"
        "PICT0001.JPG,2024-03-15 09:30:00,SiteA\n"
        "PICT0001.JPG,2024-06-11 19:45:00,SiteB\n"))
    assert metadata.lookup("PICT0001.JPG", "SiteA").month == 3
    assert metadata.lookup("PICT0001.JPG", "SiteB").month == 6
    # Without the camera the name means two different things, so it answers
    # neither rather than picking one.
    assert metadata.lookup("PICT0001.JPG") is None
    assert any("more than one capture time" in problem
               for problem in metadata.report.problems)


def test_metadata_csv_species_column_becomes_a_label():
    metadata = MetadataFile.load(csv_upload(
        "filename,timestamp,species\na.jpg,2010-07-20 06:14:06,Zebra\n"))
    assert metadata.has_labels
    assert metadata.label_for("a.jpg") == "zebra"


# ── the file name ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("IMG_20240315_093000.jpg", datetime(2024, 3, 15, 9, 30, 0)),
    ("2024-03-15_09-30-00.JPG", datetime(2024, 3, 15, 9, 30, 0)),
    ("PICT_2024-03-15_09h30m00.jpg", datetime(2024, 3, 15, 9, 30, 0)),
    ("CAM1_20241231_235959.jpg", datetime(2024, 12, 31, 23, 59, 59)),
    ("20240315_0930.jpg", datetime(2024, 3, 15, 9, 30, 0)),
])
def test_filename_patterns_camera_traps_produce(name, expected):
    assert FilenameTimestamps().lookup(name) == expected


@pytest.mark.parametrize("name", [
    "S1_B04_R1_PICT0001.JPG",      # a sequence number, not a date
    "DSC_0042.JPG",
    "photo.jpg",
    "20241332_250000.jpg",          # month 13, hour 25
])
def test_filename_parsing_declines_names_that_carry_no_time(name):
    assert FilenameTimestamps().lookup(name) is None


def test_a_date_only_name_is_refused_by_default_and_flagged_when_allowed():
    """Filling the hour in with midnight would read as nocturnal."""
    assert FilenameTimestamps().lookup("20240315.jpg") is None

    permissive = FilenameTimestamps(accept_date_only=True)
    assert permissive.lookup("20240315.jpg") == datetime(2024, 3, 15)
    assert permissive.date_only_used == 1


def test_date_only_resolutions_are_reported_under_their_own_source():
    resolver = TimestampResolver(use_exif=False, accept_date_only=True)
    assert resolver.resolve("20240315.jpg").source == FILENAME_DATE
    assert resolver.resolve("IMG_20240315_093000.jpg").source == FILENAME
    assert any("recorded at midnight" in warning
               for warning in resolver.warnings())


# ── EXIF, and what must never be used ────────────────────────────────────────

def test_exif_is_read_when_present(tmp_path):
    path = tmp_path / "shot.jpg"
    image = Image.new("RGB", (800, 600), (90, 120, 70))
    exif = image.getexif()
    exif[36867] = "2010:07:20 06:14:06"            # DateTimeOriginal
    image.save(path, "JPEG", exif=exif)

    resolver = TimestampResolver()
    resolution = resolver.resolve(path.name, "", path)
    assert resolution.timestamp == MOMENT
    assert resolution.source == EXIF


def test_file_modification_time_is_never_used(tmp_path):
    """Copying a folder resets mtime, so it is present and wrong.

    The photograph here has no EXIF, an opaque name and no supplied metadata,
    but it does have a modification time set to a plausible-looking moment.
    Resolution must still come back empty: a wrong timestamp is worse than a
    missing one, because a missing one is reported and a wrong one is counted.
    """
    path = tmp_path / "PICT0001.JPG"
    Image.new("RGB", (800, 600), (90, 120, 70)).save(path, "JPEG")
    stamp = time.mktime(MOMENT.timetuple())
    os.utime(path, (stamp, stamp))
    assert datetime.fromtimestamp(path.stat().st_mtime) == MOMENT

    resolution = TimestampResolver().resolve(path.name, "", path)
    assert not resolution.found
    assert resolution.source == NONE


# ── the chain ────────────────────────────────────────────────────────────────

def test_sources_are_tried_in_priority_order(tmp_path):
    """Annotation beats metadata beats EXIF beats the file name."""
    path = tmp_path / "IMG_20240315_093000.jpg"
    image = Image.new("RGB", (800, 600), (90, 120, 70))
    exif = image.getexif()
    exif[36867] = "2015:05:05 05:05:05"
    image.save(path, "JPEG", exif=exif)
    name = path.name

    metadata = MetadataFile.load(
        csv_upload(f"filename,timestamp\n{name},2020-02-02 02:02:02\n"))
    annotations = AnnotationFile({
        "images": [{"id": "x", "file_name": name,
                    "datetime": "2010-07-20 06:14:06"}]})

    # The full chain: the annotation file wins.
    full = TimestampResolver(annotations=annotations, metadata=metadata)
    assert full.resolve(name, "", path).source == ANNOTATIONS
    assert full.resolve(name, "", path).timestamp.year == 2010

    # Without it, the metadata file does.
    assert TimestampResolver(metadata=metadata).resolve(
        name, "", path).source == METADATA

    # Without either, EXIF — which beats the 2024 in the file name.
    exif_only = TimestampResolver().resolve(name, "", path)
    assert exif_only.source == EXIF and exif_only.timestamp.year == 2015

    # And with nothing on the file at all, the name.
    bare = tmp_path / "IMG_20240315_093000_bare.jpg"
    Image.new("RGB", (800, 600), (0, 0, 0)).save(bare, "JPEG")
    assert TimestampResolver().resolve(bare.name, "", bare).source == FILENAME


def test_the_chain_is_described_in_the_order_it_will_be_tried():
    resolver = TimestampResolver(
        annotations=AnnotationFile({
            "images": [{"id": "x", "file_name": "a.jpg",
                        "datetime": "2010-07-20 06:14:06"}]}),
        metadata=MetadataFile.load(
            csv_upload("filename,timestamp\nb.jpg,2010-07-20 06:14:06\n")))
    assert resolver.chain == [ANNOTATIONS, METADATA, EXIF, FILENAME]
    assert "annotation file → metadata file → EXIF → file name" in \
        resolver.describe_chain()


def test_an_unusable_file_is_left_out_of_the_chain_not_fatal():
    resolver = build_resolver(annotations=upload(b"{ broken", "bad.json"),
                               metadata=csv_upload("nothing,useful\n1,2\n"))
    assert resolver.chain == [EXIF, FILENAME]
    assert [report.usable for report in resolver.source_reports()] == \
        [False, False]
    # And the problems survive to be shown to the user.
    assert all(report.problems for report in resolver.source_reports())


def test_the_resolver_reports_where_every_timestamp_came_from():
    resolver = TimestampResolver(
        metadata=MetadataFile.load(
            csv_upload("filename,timestamp\nfromcsv.jpg,2010-07-20 06:14:06\n")),
        use_exif=False)
    resolver.resolve("fromcsv.jpg")
    resolver.resolve("IMG_20240315_093000.jpg")
    resolver.resolve("mystery.jpg")

    assert dict(resolver.breakdown()) == {METADATA: 1, FILENAME: 1, NONE: 1}
    # Reported in chain order, so the account reads top-down.
    assert [name for name, _ in resolver.breakdown()] == \
        [METADATA, FILENAME, NONE]
    assert "1 from metadata file" in resolver.breakdown_line()
    assert "1 with no capture time at all" in resolver.breakdown_line()


def test_labels_come_from_whichever_supplied_file_has_them():
    resolver = TimestampResolver(
        annotations=AnnotationFile(serengeti_document()),
        metadata=MetadataFile.load(csv_upload(
            "filename,timestamp,species\nown.jpg,2010-07-20 06:14:06,buffalo\n")))
    assert resolver.has_labels
    assert resolver.label_for("S1_B04_R1_PICT0001.JPG") == "zebra"
    assert resolver.label_for("own.jpg") == "buffalo"
    assert resolver.label_for("unknown.jpg") is None


def test_no_supplied_files_means_the_behaviour_ingestion_already_had():
    resolver = TimestampResolver()
    assert resolver.chain == [EXIF, FILENAME]
    assert not resolver.has_labels
    assert resolver.source_reports() == []


# ── through ingestion ────────────────────────────────────────────────────────

@pytest.fixture
def photos(tmp_path) -> Path:
    """Two cameras: one with dated names, one with opaque ones."""
    root = tmp_path / "photos"
    (root / "SiteA").mkdir(parents=True)
    (root / "SiteB").mkdir(parents=True)
    for name in ["IMG_20240315_093000.jpg", "IMG_20240418_223000.jpg"]:
        Image.new("RGB", (800, 600), (90, 120, 70)).save(
            root / "SiteA" / name, "JPEG")
    for name in ["PICT0001.JPG", "PICT0002.JPG"]:
        Image.new("RGB", (800, 600), (70, 90, 110)).save(
            root / "SiteB" / name, "JPEG")
    return root


@pytest.fixture
def registry() -> CameraRegistry:
    return CameraRegistry.from_entries([
        {"camera_id": "SiteA", "latitude": -2.15, "longitude": 34.80,
         "habitat_type": "woodland"},
        {"camera_id": "SiteB", "latitude": -2.21, "longitude": 34.85,
         "habitat_type": "open_grassland"},
    ])


def stored(db_path):
    """image_id -> (captured_at, ground truth species) straight from the DB."""
    from wildlife_monitor.db.connection import connect
    connection = connect(db_path)
    try:
        return {row["image_id"]: (row["captured_at"], row["gt"])
                for row in connection.execute(
                    "SELECT i.image_id, i.captured_at, "
                    "       COALESCE(s.canonical_name, '') AS gt "
                    "FROM Image i "
                    "LEFT JOIN Species s ON s.species_id = i.ground_truth_id")}
    finally:
        connection.close()


def test_ingestion_without_metadata_falls_back_to_the_file_name(
        photos, registry, tmp_path):
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    report = ImageIngestor(images_dir=tmp_path / "store",
                           db_path=db_path).ingest(photos, registry)

    assert report.ingested == 4
    assert report.undated == 2                      # the two PICT files
    assert dict(report.timestamp_sources) == {FILENAME: 2}
    rows = stored(db_path)
    assert rows["SiteA_IMG_20240315_093000"][0] == "2024-03-15 09:30:00"
    assert rows["SiteB_PICT0001"][0] == ""


def test_a_metadata_file_recovers_the_photographs_with_opaque_names(
        photos, registry, tmp_path):
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    resolver = build_resolver(metadata=csv_upload(
        "filename,timestamp,camera\n"
        "PICT0001.JPG,2024-05-01 07:10:00,SiteB\n"
        "PICT0002.JPG,2024-06-11 19:45:00,SiteB\n"))
    report = ImageIngestor(images_dir=tmp_path / "store", db_path=db_path,
                           resolver=resolver).ingest(photos, registry)

    assert report.undated == 0
    assert dict(report.timestamp_sources) == {METADATA: 2, FILENAME: 2}
    rows = stored(db_path)
    assert rows["SiteB_PICT0001"][0] == "2024-05-01 07:10:00"
    assert report.first_capture.startswith("2024-03-15")
    assert report.last_capture.startswith("2024-06-11")


def test_an_annotation_file_supplies_times_and_ground_truth_through_ingestion(
        photos, registry, tmp_path):
    """The Serengeti route, generalised: labels turn accuracy measurable."""
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    resolver = build_resolver(annotations=json_upload({
        "categories": [{"id": 1, "name": "zebra"}],
        "images": [{"id": "one", "file_name": "SiteB/PICT0001.JPG",
                    "location": "SiteB", "datetime": "2010-07-20 06:14:06"}],
        "annotations": [{"image_id": "one", "category_id": 1,
                         "datetime": "2010-07-20 06:14:06"}]}))
    report = ImageIngestor(images_dir=tmp_path / "store", db_path=db_path,
                           resolver=resolver).ingest(photos, registry)

    assert report.labelled == 1
    assert dict(report.timestamp_sources) == {ANNOTATIONS: 1, FILENAME: 2}
    captured, ground_truth = stored(db_path)["SiteB_PICT0001"]
    assert captured == "2010-07-20 06:14:06"
    assert ground_truth == "zebra"


def test_the_annotation_file_outranks_the_file_name_during_ingestion(
        photos, registry, tmp_path):
    """A curated record beats a name, even when the name parses cleanly."""
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    resolver = build_resolver(annotations=json_upload({
        "images": [{"id": "a", "file_name": "IMG_20240315_093000.jpg",
                    "datetime": "2010-07-20 06:14:06"}]}))
    ImageIngestor(images_dir=tmp_path / "store", db_path=db_path,
                  resolver=resolver).ingest(photos, registry)

    assert stored(db_path)["SiteA_IMG_20240315_093000"][0] == \
        "2010-07-20 06:14:06"


def test_a_dry_run_reports_the_sources_without_writing(
        photos, registry, tmp_path):
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    resolver = build_resolver(metadata=csv_upload(
        "filename,timestamp,species\nPICT0001.JPG,2024-05-01 07:10:00,zebra\n"))
    report = ImageIngestor(images_dir=tmp_path / "store", db_path=db_path,
                           resolver=resolver).ingest(
        photos, registry, dry_run=True)

    assert report.ingested == 4
    assert dict(report.timestamp_sources) == {METADATA: 1, FILENAME: 2}
    assert report.labelled == 1
    assert stored(db_path) == {}                   # nothing was written


def test_uploads_use_the_same_chain_as_folder_ingestion(registry, tmp_path):
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    resolver = build_resolver(metadata=csv_upload(
        "filename,timestamp\nPICT0001.JPG,2024-05-01 07:10:00\n"))

    buffer = io.BytesIO()
    Image.new("RGB", (800, 600), (90, 120, 70)).save(buffer, "JPEG")
    files = [upload(buffer.getvalue(), "PICT0001.JPG"),
             upload(buffer.getvalue(), "IMG_20240418_223000.jpg"),
             upload(buffer.getvalue(), "mystery.JPG")]

    report = ImageIngestor(images_dir=tmp_path / "store", db_path=db_path,
                           resolver=resolver).ingest_uploads(
        files, "SiteB", registry)

    assert report.ingested == 3
    assert report.undated == 1
    assert dict(report.timestamp_sources) == {METADATA: 1, FILENAME: 1}


def test_the_ingestion_report_carries_the_chain_and_its_warnings(
        photos, registry, tmp_path):
    db_path = tmp_path / "wildlife.db"
    init_db(db_path)
    report = ImageIngestor(images_dir=tmp_path / "store",
                           db_path=db_path).ingest(photos, registry)

    assert "EXIF → file name" in report.timestamp_chain
    assert "2 from file name" in report.timestamp_line()
    assert any("no capture time from any source" in note
               for note in report.timestamp_notes)
