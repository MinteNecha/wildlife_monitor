"""Upload page (UC1, FR1) — batch ingestion with validation."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard.components import header, rule, note

MIN_WIDTH, MIN_HEIGHT = 640, 480
_ACCEPTED_FORMATS = {"JPEG", "PNG"}


def render(species: str) -> None:
    header("Upload Camera Trap Images",
           "Batch ingestion with format and metadata validation · UC1")
    st.caption(f"Accepted: JPEG, PNG · Minimum resolution "
               f"{MIN_WIDTH}×{MIN_HEIGHT} · EXIF timestamps read when present.")

    files = st.file_uploader("Camera trap images",
                             type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if not files:
        st.info("Select one or more images. Validation runs on upload.")
        return

    accepted, rejected = _validate(files)
    _summary_metrics(files, accepted, rejected)
    rule()
    _results_tables(accepted, rejected)
    if accepted:
        rule()
        _preview(files)
    st.button(f"Ingest {len(accepted)} images", disabled=not accepted)


def _validate(files) -> tuple[list[dict], list[tuple[str, str]]]:
    accepted: list[dict] = []
    rejected: list[tuple[str, str]] = []
    for file in files:
        try:
            image = Image.open(file)
            width, height = image.size
            fmt = (image.format or "?").upper()
            if fmt not in _ACCEPTED_FORMATS:
                rejected.append((file.name, f"Unsupported format: {fmt}"))
            elif width < MIN_WIDTH or height < MIN_HEIGHT:
                rejected.append((file.name,
                                 f"Resolution {width}×{height} below minimum"))
            else:
                exif = image.getexif()
                timestamp = exif.get(36867) or exif.get(306)
                accepted.append({
                    "File": file.name, "Format": fmt,
                    "Resolution": f"{width}×{height}",
                    "Timestamp": timestamp or "not in EXIF",
                    "Size (KB)": round(file.size / 1024, 1)})
        except Exception as exc:
            rejected.append((file.name, f"Unreadable file: {exc}"))
    return accepted, rejected


def _summary_metrics(files, accepted, rejected) -> None:
    missing = sum(row["Timestamp"] == "not in EXIF" for row in accepted)
    for column, (label, value) in zip(st.columns(4), [
        ("Received", len(files)), ("Accepted", len(accepted)),
        ("Rejected", len(rejected)), ("No Timestamp", missing),
    ]):
        column.metric(label, value)


def _results_tables(accepted, rejected) -> None:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Accepted Files")
        if accepted:
            st.dataframe(pd.DataFrame(accepted), width="stretch",
                         height=240, hide_index=True)
        else:
            st.warning("No files passed validation.")
    with right:
        st.subheader("Rejected Files")
        if rejected:
            for name, reason in rejected:
                note(f"<b>{name}</b><br>{reason}", ok=False)
        else:
            note("All files passed validation.", ok=True)


def _preview(files) -> None:
    st.subheader("Preview")
    columns = st.columns(4)
    for column, file in zip(columns, files[:4]):
        with column:
            try:
                st.image(Image.open(file), width="stretch")
                st.caption(file.name[:28])
            except Exception:
                pass
