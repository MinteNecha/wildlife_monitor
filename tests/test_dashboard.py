"""
Every dashboard page renders, and the Upload page's steps work end to end.

These run the real Streamlit script headlessly through ``AppTest``. They exist
because a page module can break in ways the rest of the suite cannot see — an
import that pulls in torch, a theme key passed twice, a column that no longer
exists — and none of that shows up until someone opens the page.

Navigation is selected by setting the sidebar radio, not by writing to
``st.session_state``: the radio has no key, so a session-state entry is
ignored and every page would silently render as Overview. A test that appears
to check nine pages while checking one is worse than no test, so the first
assertion here is that the page actually changed.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

# AppTest resolves a relative path against the calling file, so give it an
# absolute one rather than depending on where pytest was invoked from.
APP = str(Path(__file__).resolve().parents[1]
          / "wildlife_monitor" / "dashboard" / "app.py")
PAGES = ["Overview", "Upload", "Species Detection", "Detection Map",
         "Behavioural Analysis", "Pipeline Comparison", "Image Review",
         "Settings", "Export"]


def jpeg(width: int = 800, height: int = 600) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 70)).save(buffer, "JPEG")
    return buffer.getvalue()


def open_page(page: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=120)
    app.run()
    app.radio(key="page").set_value(page).run()
    assert app.radio(key="page").value == page, "navigation did not change page"
    return app


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_raising(page):
    app = open_page(page)
    assert not app.exception, f"{page}: {app.exception}"


def test_the_upload_page_offers_all_four_steps():
    app = open_page("Upload")
    headings = [heading.value for heading in app.subheader]
    assert headings[:3] == ["1 · Camera", "2 · Capture Times",
                            "3 · Photographs"]
    # Step 4 appears once photographs are in hand; step 2's two uploaders and
    # the image uploader are present from the start.
    labels = [uploader.label for uploader in app.file_uploader]
    assert labels == ["Metadata CSV", "Annotation JSON", "Camera trap images"]


def test_supplied_metadata_resolves_times_the_photographs_do_not_carry():
    """The end-to-end claim of the timestamp chain, through the real page.

    Three photographs: one named after nothing, one whose name carries a
    date and time, and one covered by an uploaded annotation file. Each must
    come back with its capture time attributed to the right source, and the
    photograph no source knows about must come back empty rather than guessed.
    """
    app = open_page("Upload")

    app.file_uploader(key="timestamp_csv").set_value([(
        "times.csv",
        b"filename,timestamp,species\nPICT0001.JPG,2024-03-20 07:10:00,zebra\n",
        "text/csv")]).run()
    app.file_uploader(key="timestamp_json").set_value([(
        "annotations.json",
        json.dumps({
            "categories": [{"id": 1, "name": "wildebeest"}],
            "images": [{"id": "A1", "file_name": "PICT0009.JPG",
                        "datetime": "2010-07-20 06:14:06"}],
            "annotations": [{"image_id": "A1", "category_id": 1}],
        }).encode(),
        "application/json")]).run()
    assert not app.exception

    images = [uploader for uploader in app.file_uploader
              if uploader.accept_multiple_files][0]
    images.set_value([("PICT0001.JPG", jpeg(), "image/jpeg"),
                      ("IMG_20240418_223000.jpg", jpeg(), "image/jpeg"),
                      ("PICT0009.JPG", jpeg(), "image/jpeg"),
                      ("mystery.JPG", jpeg(), "image/jpeg")]).run()
    assert not app.exception

    table = app.dataframe[0].value.set_index("File")
    assert table.loc["PICT0001.JPG", "From"] == "metadata file"
    assert table.loc["PICT0001.JPG", "Timestamp"] == "2024-03-20 07:10:00"
    assert table.loc["IMG_20240418_223000.jpg", "From"] == "file name"
    assert table.loc["PICT0009.JPG", "From"] == "annotation file"
    assert table.loc["mystery.JPG", "Timestamp"] == "none found"

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Received"] == "4"
    assert metrics["No Timestamp"] == "1"


def test_a_broken_metadata_file_is_reported_and_the_page_carries_on():
    app = open_page("Upload")
    app.file_uploader(key="timestamp_json").set_value([(
        "annotations.json", b"{ not json", "application/json")]).run()

    assert not app.exception
    captions = " ".join(caption.value for caption in app.caption)
    assert "not valid JSON" in captions
    # The chain drops the unusable file rather than refusing to continue.
    assert "EXIF → file name" in captions
