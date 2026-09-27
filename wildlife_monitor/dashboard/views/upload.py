"""
Upload page (UC1, FR1) — camera setup, validation, and ingestion.

Three steps, in the order a user actually needs them:

    1. say which camera these photographs came from, and where it is
    2. upload the photographs and see what passed
    3. ingest them

Step 1 exists because a browser upload carries no folder, so the camera has
to be stated. It is filled in on the page rather than in a spreadsheet, and
the system writes the file, which removes any chance of getting the format
wrong. A file prepared elsewhere can be uploaded instead.

Where images fail the resolution check the page offers three honest choices
rather than one silent fix: replace them, lower the requirement, or let the
system enlarge them — with enlargement labelled for what it is, since it adds
no detail and every enlarged image is flagged in the database afterwards.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule, note
from wildlife_monitor.dashboard.theme import BLACK, GREY
from wildlife_monitor.data.cameras import (
    CameraRegistry, blocking_issues, warnings,
)
from wildlife_monitor.data.ingestion import ImageIngestor
from wildlife_monitor.data.preparation import ImagePreparer, describe_options
from wildlife_monitor.data.sufficiency import MIN_SPAN_MONTHS
from wildlife_monitor.data.validator import ImageValidator
from wildlife_monitor.db import init_db
from wildlife_monitor.pipeline2.feature_extractor import HABITAT_CLASSES

CAMERA_FILE = "data/cameras.csv"


def render(species: str) -> None:
    header("Upload Camera Trap Images",
           "Camera setup, validation and ingestion · UC1, FR1")

    camera_id, registry = _camera_step()
    rule()

    min_width = int(st.session_state.get("min_width", 640))
    min_height = int(st.session_state.get("min_height", 480))
    validator = ImageValidator(min_width=min_width, min_height=min_height)

    st.subheader("2 · Photographs")
    st.caption(validator.requirements)
    files = st.file_uploader("Camera trap images",
                             type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
                             accept_multiple_files=True)
    if not files:
        st.info("Select one or more images. Validation runs on upload.")
        return

    accepted, rejected = validator.validate_batch(files)
    _summary_metrics(files, accepted, rejected)
    rule()
    _coverage_summary(accepted)
    rule()
    _results_tables(accepted, rejected)

    preparer = None
    if rejected:
        rule()
        preparer = _recovery_options(rejected, min_width, min_height)

    rule()
    _ingest_step(files, camera_id, registry, validator, preparer)


# ── Step 1: the camera ───────────────────────────────────────────────────────

def _camera_step() -> tuple[str, CameraRegistry]:
    st.subheader("1 · Camera")
    st.caption("A browser upload carries no folder, so tell the system which "
               "camera these photographs came from. For many cameras at once, "
               "use scripts/ingest_images.py instead.")

    registry = _load_registry()
    existing = registry.camera_ids

    source = st.radio(
        "Camera details",
        ["Use a saved camera", "Add a camera", "Upload a camera file"],
        horizontal=True,
        index=0 if existing else 1)

    if source == "Upload a camera file":
        return _upload_registry(registry)
    if source == "Add a camera" or not existing:
        return _add_camera(registry)

    camera_id = st.selectbox("Camera", existing)
    details = registry.lookup().get(camera_id, {})
    st.markdown(
        f"<div style='font-size:12px;color:{GREY}'>"
        f"{details.get('latitude')}, {details.get('longitude')} · "
        f"{details.get('habitat_type', 'unknown')}</div>",
        unsafe_allow_html=True)
    return camera_id, registry


def _load_registry() -> CameraRegistry:
    from pathlib import Path
    path = Path(CAMERA_FILE)
    if path.exists():
        try:
            return CameraRegistry.load(path)
        except Exception:
            pass
    return CameraRegistry()


def _add_camera(registry: CameraRegistry) -> tuple[str, CameraRegistry]:
    """Type the details; the system writes the file in the right format."""
    left, right = st.columns(2)
    with left:
        camera_id = st.text_input("Camera name", placeholder="SiteA")
        habitat = st.selectbox("Habitat", HABITAT_CLASSES)
    with right:
        latitude = st.number_input("Latitude", -90.0, 90.0, value=-2.3333,
                                    format="%.4f")
        longitude = st.number_input("Longitude", -180.0, 180.0, value=34.8333,
                                     format="%.4f")

    if not camera_id.strip():
        st.info("Enter a camera name to continue.")
        return "", registry

    entries = registry.normalised().to_dict("records") if not registry.frame.empty else []
    entries = [row for row in entries if row["camera_id"] != camera_id.strip()]
    entries.append({"camera_id": camera_id.strip(), "latitude": latitude,
                    "longitude": longitude, "habitat_type": habitat})
    updated = CameraRegistry.from_entries(entries)

    issues = updated.validate()
    for issue in blocking_issues(issues):
        note(issue.message, ok=False)
    if blocking_issues(issues):
        return "", registry

    columns = st.columns([1, 1, 2])
    if columns[0].button("Save camera"):
        try:
            path = updated.write(CAMERA_FILE)
            st.success(f"Saved to {path}")
        except Exception as error:
            st.error(f"Could not save: {error}")
    columns[1].download_button("Download camera file",
                                updated.to_csv_bytes(), "cameras.csv",
                                "text/csv")
    return camera_id.strip(), updated


def _upload_registry(registry: CameraRegistry) -> tuple[str, CameraRegistry]:
    """Accept a camera file the user prepared themselves."""
    uploaded = st.file_uploader("Camera file (CSV)", type=["csv"],
                                key="camera_csv")
    if uploaded is None:
        st.caption("Expected columns: camera_id, latitude, longitude, "
                   "habitat_type. Common alternatives such as site, lat and "
                   "lon are also understood.")
        return "", registry

    try:
        loaded = CameraRegistry.load(uploaded)
    except Exception as error:
        st.error(f"Could not read that file: {error}")
        return "", registry

    issues = loaded.validate()
    for issue in warnings(issues):
        note(issue.message, ok=False)
    blocking = blocking_issues(issues)
    for issue in blocking:
        note(issue.message, ok=False)
    if blocking:
        return "", loaded

    st.dataframe(loaded.normalised(), width="stretch", hide_index=True)
    if st.button("Save camera file"):
        loaded.write(CAMERA_FILE)
        st.success(f"Saved to {CAMERA_FILE}")

    identifiers = loaded.camera_ids
    camera_id = st.selectbox("Camera for this upload", identifiers) if identifiers else ""
    return camera_id, loaded


# ── Step 2: validation feedback ──────────────────────────────────────────────

def _summary_metrics(files, accepted, rejected) -> None:
    missing = sum(not result.details.get("has_timestamp") for result in accepted)
    for column, (label, value) in zip(st.columns(4), [
        ("Received", len(files)), ("Accepted", len(accepted)),
        ("Rejected", len(rejected)), ("No Timestamp", missing),
    ]):
        column.metric(label, value)


def _coverage_summary(accepted) -> None:
    """Date range covered, and what that span can support.

    Time span is the constraint most likely to invalidate a behavioural
    result, and it is knowable from EXIF before any detection has run.
    """
    st.subheader("Coverage")
    stamps = sorted(result.details.get("timestamp", "") for result in accepted
                    if result.details.get("has_timestamp"))

    if not stamps:
        note("None of these images carry an EXIF capture time. Without "
             "timestamps the system cannot order detections, so they can be "
             "classified but not used for behavioural analysis.", ok=False)
        return

    first, last = stamps[0][:10], stamps[-1][:10]
    span_months = (pd.to_datetime(last) - pd.to_datetime(first)).days / 30.44

    columns = st.columns(3)
    columns[0].metric("Earliest", first)
    columns[1].metric("Latest", last)
    columns[2].metric("Span", f"{span_months:.1f} months")

    if span_months >= MIN_SPAN_MONTHS[1]:
        note(f"This batch spans {span_months:.1f} months — enough coverage "
             f"for seasonal movement analysis once detections have run.",
             ok=True)
    elif span_months >= MIN_SPAN_MONTHS[0]:
        note(f"This batch spans {span_months:.1f} months. Activity timing "
             f"will be reliable. Movement classification will work but should "
             f"be read as indicative — {MIN_SPAN_MONTHS[1]} months or more "
             f"makes it dependable.", ok=False)
    else:
        note(f"This batch spans {span_months:.1f} months. Activity timing "
             f"(day/night) will work. Movement classification will not: "
             f"migratory and territorial differ by how presence is spread "
             f"across seasons, and under {MIN_SPAN_MONTHS[0]} months every "
             f"camera looks alike. More photographs over the same window will "
             f"not change this — more months will.", ok=False)


def _results_tables(accepted, rejected) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Accepted Files")
        if accepted:
            st.dataframe(pd.DataFrame([{
                "File": result.details.get("name", ""),
                "Format": result.details.get("format", ""),
                "Resolution": result.details.get("resolution", ""),
                "Timestamp": result.details.get("timestamp") or "not in EXIF",
            } for result in accepted]), width="stretch", height=240,
                hide_index=True)
        else:
            st.warning("No files passed validation.")
    with right:
        st.subheader("Rejected Files")
        if rejected:
            for result in rejected:
                note(f"<b>{result.details.get('name', 'file')}</b><br>"
                     f"{result.reason}", ok=False)
        else:
            note("All files passed validation.", ok=True)


# ── Recovery options ─────────────────────────────────────────────────────────

def _recovery_options(rejected, min_width: int,
                      min_height: int) -> ImagePreparer | None:
    """Offer the honest ways to deal with images that failed the check."""
    st.subheader("Rejected Images — Your Options")

    too_small = [result for result in rejected
                 if "Resolution" in (result.reason or "")]
    wrong_format = [result for result in rejected
                    if "format" in (result.reason or "").lower()]

    if wrong_format and not too_small:
        note(f"{len(wrong_format)} file(s) are in a format the system does "
             f"not read. Converting them to JPEG loses nothing.", ok=True)
        if st.checkbox("Convert these to JPEG", value=True):
            return ImagePreparer(min_width=min_width, min_height=min_height)
        return None

    for title, explanation in describe_options(min_width, min_height):
        st.markdown(
            f"<div style='font-size:12px;color:{BLACK};font-weight:700;"
            f"margin-top:6px'>{title}</div>"
            f"<div style='font-size:11px;color:{GREY}'>{explanation}</div>",
            unsafe_allow_html=True)

    st.markdown("")
    choice = st.radio(
        "How should these be handled?",
        ["Leave them out", "Lower the requirement", "Enlarge them"],
        horizontal=True)

    if choice == "Lower the requirement":
        left, right = st.columns(2)
        st.session_state["min_width"] = left.number_input(
            "Minimum width", 64, 4096, min_width, 32)
        st.session_state["min_height"] = right.number_input(
            "Minimum height", 64, 4096, min_height, 32)
        note("The images will be accepted at their true resolution, and "
             "recorded as such. Nothing is altered.", ok=True)
        st.caption("Change the values above, then re-check the file list.")
        return ImagePreparer(min_width=min_width, min_height=min_height)

    if choice == "Enlarge them":
        note("Enlarging lets these images pass the check but adds no detail — "
             "detection will be no better than at their original size. Every "
             "enlarged image is flagged in the database so its results stay "
             "identifiable.", ok=False)
        return ImagePreparer(min_width=min_width, min_height=min_height,
                              allow_upscale=True)

    return None


# ── Step 3: ingestion ────────────────────────────────────────────────────────

def _ingest_step(files, camera_id: str, registry: CameraRegistry,
                 validator: ImageValidator,
                 preparer: ImagePreparer | None) -> None:
    st.subheader("3 · Ingest")

    if not camera_id:
        st.info("Choose or add a camera in step 1 before ingesting.")
        return

    if preparer is not None:
        st.caption(f"Preparation enabled: {preparer.description}")

    if not st.button(f"Ingest {len(files)} images to {camera_id}",
                     type="primary"):
        return

    try:
        init_db()
        for file in files:
            file.seek(0)
        report = ImageIngestor(validator, preparer).ingest_uploads(
            files, camera_id, registry)
    except Exception as error:
        st.error(f"Ingestion failed: {error}")
        return

    st.success(report.summary_line())
    if report.upscaled:
        note(f"{report.upscaled} image(s) were enlarged and are flagged as "
             f"such. Their detections will be no better than at the original "
             f"size.", ok=False)
    if report.undated:
        note(f"{report.undated} image(s) had no capture time and cannot be "
             f"used for behavioural analysis.", ok=False)
    if report.rejected:
        for name, reason in report.rejected[:5]:
            note(f"<b>{name}</b><br>{reason}", ok=False)

    if report.ingested:
        st.markdown("**Next:** identify the species in these images.")
        st.code("python scripts/classify_images.py --all", language="bash")


def _preview(files) -> None:
    st.subheader("Preview")
    for column, file in zip(st.columns(4), files[:4]):
        with column:
            try:
                file.seek(0)
                st.image(Image.open(file), width="stretch")
                st.caption(file.name[:28])
            except Exception:
                pass
