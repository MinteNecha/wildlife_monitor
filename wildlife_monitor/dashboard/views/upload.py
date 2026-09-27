"""
Upload page (UC1, FR1) — camera setup, capture times, validation, ingestion.

Four steps, in the order a user actually needs them:

    1. say which camera these photographs came from, and where it is
    2. optionally supply capture times, if the photographs have lost theirs
    3. upload the photographs and see what passed
    4. ingest them

Step 1 exists because a browser upload carries no folder, so the camera has
to be stated. It is filled in on the page rather than in a spreadsheet, and
the system writes the file, which removes any chance of getting the format
wrong. A file prepared elsewhere can be uploaded instead.

Step 2 exists because EXIF is routinely stripped — by editing software, by
messaging apps, by anything that re-saves a JPEG — and a photograph with no
capture time contributes nothing to behavioural analysis. A metadata CSV or an
annotation JSON puts the times back. The coverage figures below are computed
from the resolved times, not from EXIF alone, so the span shown is the span
that will actually be analysed.

Where images fail the resolution check the page offers three honest choices
rather than one silent fix: replace them, lower the requirement, or let the
system enlarge them — with enlargement labelled for what it is, since it adds
no detail and every enlarged image is flagged in the database afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard.components import header, rule, note
from wildlife_monitor.dashboard.theme import BLACK, GREY
from wildlife_monitor.data.cameras import (
    CameraRegistry, blocking_issues, warnings,
)
from wildlife_monitor.data.ingestion import ImageIngestor
from wildlife_monitor.data.preparation import ImagePreparer, describe_options
from wildlife_monitor.data.sufficiency import MIN_SPAN_MONTHS
from wildlife_monitor.data import timestamps
from wildlife_monitor.data.timestamps import (
    AnnotationFile, MetadataFile, TimestampResolver,
)
from wildlife_monitor.data.validator import ImageValidator
from wildlife_monitor.db import init_db
from wildlife_monitor.pipeline2.feature_extractor import HABITAT_CLASSES

CAMERA_FILE = "data/cameras.csv"


def render(species: str) -> None:
    header("Upload Camera Trap Images",
           "Camera setup, validation and ingestion · UC1, FR1")

    camera_id, registry = _camera_step()
    rule()

    sources = _timestamp_step()
    rule()

    min_width = int(st.session_state.get("min_width", 640))
    min_height = int(st.session_state.get("min_height", 480))
    validator = ImageValidator(min_width=min_width, min_height=min_height)

    st.subheader("3 · Photographs")
    st.caption(validator.requirements)
    files = st.file_uploader("Camera trap images",
                             type=["jpg", "jpeg", "png", "tif", "tiff", "webp"],
                             accept_multiple_files=True)
    if not files:
        st.info("Select one or more images. Validation runs on upload.")
        return

    accepted, rejected = validator.validate_batch(files)

    # A separate resolver for the preview, so the counts it accumulates are
    # not carried into the ingestion run's own report.
    preview = _build_resolver(sources, validator)
    resolved = {getattr(file, "name", ""):
                preview.resolve(getattr(file, "name", ""), camera_id, file)
                for file in files}
    for file in files:
        file.seek(0)

    _summary_metrics(files, accepted, rejected, resolved)
    rule()
    _coverage_summary(accepted, resolved, preview)
    rule()
    _results_tables(accepted, rejected, resolved)

    preparer = None
    if rejected:
        rule()
        preparer = _recovery_options(rejected, min_width, min_height)

    rule()
    _ingest_step(files, camera_id, registry, validator, preparer, sources)


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


# ── Step 2: capture times ────────────────────────────────────────────────────

@dataclass
class _Sources:
    """The optional timestamp files this upload will use."""

    annotations: AnnotationFile | None = None
    metadata: MetadataFile | None = None
    accept_date_only: bool = False


def _build_resolver(sources: _Sources,
                    validator: ImageValidator) -> TimestampResolver:
    """A fresh resolver over the same loaded files.

    Each resolver keeps its own tally of which source answered, so the preview
    pass and the ingestion run get separate instances and neither inflates the
    other's report. Parsing is not repeated: the loaded files are shared.
    """
    return TimestampResolver(annotations=sources.annotations,
                             metadata=sources.metadata,
                             validator=validator,
                             accept_date_only=sources.accept_date_only)


def _cached(upload, key: str, loader):
    """Parse an uploaded file once per upload, not once per interaction.

    Streamlit re-runs this script on every widget change. An annotation file
    can be hundreds of megabytes, so re-parsing it each time would make the
    page unusable; the parsed result is kept against the file's name and size.
    """
    if upload is None:
        st.session_state.pop(key, None)
        return None
    stamp = (getattr(upload, "name", ""), getattr(upload, "size", 0))
    cached = st.session_state.get(key)
    if cached and cached[0] == stamp:
        return cached[1]
    with st.spinner(f"Reading {stamp[0]}…"):
        loaded = loader(upload)
    st.session_state[key] = (stamp, loaded)
    return loaded


def _timestamp_step() -> _Sources:
    """Offer the ways to supply capture times the photographs have lost."""
    st.subheader("2 · Capture Times")
    st.caption("Optional. Capture times are read from the photographs' own "
               "EXIF data, and from their file names where those carry a date "
               "and time. Supply a file here if yours have neither — EXIF is "
               "routinely stripped by editing and messaging software.")

    sources = _Sources()
    left, right = st.columns(2)

    with left:
        st.markdown("**Metadata CSV**")
        st.caption("Columns: filename, timestamp. A camera column is used "
                   "where file names repeat across sites, and a species "
                   "column is stored as ground truth.")
        upload = st.file_uploader("Metadata CSV", type=["csv"],
                                  key="timestamp_csv",
                                  label_visibility="collapsed")
        sources.metadata = _cached(upload, "_metadata_file", MetadataFile.load)
        if sources.metadata is not None:
            _source_note(sources.metadata.report)

    with right:
        st.markdown("**Annotation JSON**")
        st.caption("A COCO Camera Traps file such as "
                   "SnapshotSerengetiS01.json. Takes precedence over "
                   "everything else, and its species labels are stored as "
                   "ground truth.")
        upload = st.file_uploader("Annotation JSON", type=["json"],
                                  key="timestamp_json",
                                  label_visibility="collapsed")
        sources.annotations = _cached(upload, "_annotation_file",
                                       AnnotationFile.load)
        if sources.annotations is not None:
            _source_note(sources.annotations.report)

    sources.accept_date_only = st.checkbox(
        "Accept a date with no time of day from the file name",
        value=False,
        help="A name like 20240315.jpg gives a real month but no hour, so it "
             "is recorded at midnight. Seasonal and movement results stay "
             "valid; day/night activity timing does not.")

    st.caption(_build_resolver(sources, ImageValidator()).describe_chain()
               + "  File modification time is never used: copying a folder "
                 "resets it, so it is always present and almost always wrong.")
    return sources


def _source_note(report) -> None:
    """Say what a supplied file loaded, and what went wrong if anything did."""
    note(report.summary_line(), ok=report.usable)
    for problem in report.problems:
        st.caption(f"· {problem}")


# ── Step 3: validation feedback ──────────────────────────────────────────────

def _has_time(resolved: dict, file) -> bool:
    resolution = resolved.get(getattr(file, "name", ""))
    return bool(resolution is not None and resolution.found)


def _summary_metrics(files, accepted, rejected, resolved) -> None:
    missing = sum(1 for file in files if not _has_time(resolved, file))
    for column, (label, value) in zip(st.columns(4), [
        ("Received", len(files)), ("Accepted", len(accepted)),
        ("Rejected", len(rejected)), ("No Timestamp", missing),
    ]):
        column.metric(label, value)


def _coverage_summary(accepted, resolved, resolver) -> None:
    """Date range covered, and what that span can support.

    Time span is the constraint most likely to invalidate a behavioural
    result, and it is knowable before any detection has run. The span is
    computed from the resolved times rather than from EXIF alone, so a user
    who supplied a metadata file sees the coverage they will actually get.
    """
    st.subheader("Coverage")
    names = {result.details.get("name", "") for result in accepted}
    stamps = sorted(resolution.text for name, resolution in resolved.items()
                    if name in names and resolution.found)

    found = [(source, count) for source, count in resolver.breakdown()
             if source != timestamps.NONE]
    if found:
        st.caption(" · ".join(f"{count:,} from {source}"
                              for source, count in found))

    if not stamps:
        note("None of these images has a capture time, from EXIF, their file "
             "names or any file you supplied. Without timestamps the system "
             "cannot order detections, so they can be classified but not used "
             "for behavioural analysis. A metadata CSV in step 2 would "
             "recover them.", ok=False)
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


def _results_tables(accepted, rejected, resolved) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Accepted Files")
        if accepted:
            rows = []
            for result in accepted:
                name = result.details.get("name", "")
                resolution = resolved.get(name)
                rows.append({
                    "File": name,
                    "Format": result.details.get("format", ""),
                    "Resolution": result.details.get("resolution", ""),
                    "Timestamp": (resolution.text if resolution
                                  and resolution.found else "none found"),
                    "From": (resolution.source if resolution
                             and resolution.found else "—"),
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", height=240,
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


# ── Step 4: ingestion ────────────────────────────────────────────────────────

def _ingest_step(files, camera_id: str, registry: CameraRegistry,
                 validator: ImageValidator,
                 preparer: ImagePreparer | None,
                 sources: _Sources) -> None:
    st.subheader("4 · Ingest")

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
        ingestor = ImageIngestor(
            validator, preparer,
            resolver=_build_resolver(sources, validator))
        report = ingestor.ingest_uploads(files, camera_id, registry)
    except Exception as error:
        st.error(f"Ingestion failed: {error}")
        return

    st.success(report.summary_line())
    if report.timestamp_sources:
        st.caption(report.timestamp_line())
    if report.labelled:
        note(f"{report.labelled} image(s) carry a species label from the file "
             f"you supplied. These are stored as ground truth, so Image Review "
             f"will report measured accuracy for them instead of showing them "
             f"as unverified.", ok=True)
    if report.upscaled:
        note(f"{report.upscaled} image(s) were enlarged and are flagged as "
             f"such. Their detections will be no better than at the original "
             f"size.", ok=False)
    for message in report.timestamp_notes:
        note(message, ok=False)
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
