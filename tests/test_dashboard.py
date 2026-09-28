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

from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES,
)
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

def test_the_settings_page_offers_only_controls_that_take_effect():
    """Six controls used to write to session state and nowhere else.

    Pinning the widget inventory is what stops one creeping back: a new
    control here has to be wired to something before this test will pass.
    """
    app = open_page("Settings")

    assert [box.label for box in app.selectbox] == [
        "Species", "Pipeline", "Device"]        # the first two are the sidebar
    assert [slider.label for slider in app.slider] == ["Confidence threshold"]
    assert [box.label for box in app.number_input] == ["Images per run"]
    assert len(app.text_input) == 0
    assert [radio.label for radio in app.radio] == ["Navigation"]


def test_applying_saves_the_three_settings_to_config_json(tmp_path,
                                                          monkeypatch):
    from wildlife_monitor.config import settings as config_settings

    monkeypatch.setattr(config_settings, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config_settings.SystemConfig, "_instance", None)

    app = open_page("Settings")
    app.slider[0].set_value(0.65).run()
    app.button(key="apply_config").click().run()
    assert not app.exception

    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["confidence_threshold"] == pytest.approx(0.65)
    assert set(saved) == {"device", "confidence_threshold", "top_n"}


def test_the_fixed_panel_states_the_classes_the_code_actually_uses():
    """Read from labelling.py, so the page cannot drift from the taxonomy."""
    app = open_page("Settings")
    table = app.dataframe[0].value.set_index("Setting")

    assert table.loc["Activity classes", "Value"] == ", ".join(ACTIVITY_CLASSES)
    assert table.loc["Movement classes", "Value"] == ", ".join(MOVEMENT_CLASSES)
    assert table.loc["Territorial cameras", "Value"] == (
        "top 25% of cameras by site fidelity")


def test_an_out_of_range_value_is_rejected_with_its_valid_range():
    """FR9 and the UC8 alternative flow.

    The widgets clamp their own ranges, so this path is not reachable by
    clicking. It is reachable by a caller, and the message a user would see
    is the validator's own description, so that wiring is what is checked.
    """
    from wildlife_monitor.dashboard.views.settings import _validate

    errors = dict(_validate({"confidence_threshold": 5.0, "top_n": 0,
                             "device": "tpu"}))
    assert "between 0.0 and 1.0" in errors["Confidence threshold"]
    assert "between 1 and 5000" in errors["Images per run"]
    assert "cuda, cpu" in errors["Device"]
