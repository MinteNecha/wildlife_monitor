"""Upload page (UC1, FR1) — batch ingestion with validation."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard.components import header, rule, note
from wildlife_monitor.data.validator import ImageValidator

_validator = ImageValidator()


def render(species: str) -> None:
    header("Upload Camera Trap Images",
           "Batch ingestion with format and metadata validation · UC1")
    st.caption(_validator.requirements)

    files = st.file_uploader("Camera trap images",
                             type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if not files:
        st.info("Select one or more images. Validation runs on upload.")
        return

    accepted, rejected = _validator.validate_batch(files)
    _summary_metrics(files, accepted, rejected)
    rule()
    _results_tables(accepted, rejected)
    if accepted:
        rule()
        _preview(files)
    st.button(f"Ingest {len(accepted)} images", disabled=not accepted)


def _summary_metrics(files, accepted, rejected) -> None:
    missing = sum(not result.details.get("has_timestamp") for result in accepted)
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
            table = pd.DataFrame([{
                "File": result.details.get("name", ""),
                "Format": result.details.get("format", ""),
                "Resolution": result.details.get("resolution", ""),
                "Timestamp": result.details.get("timestamp") or "not in EXIF",
            } for result in accepted])
            st.dataframe(table, width="stretch", height=240, hide_index=True)
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


def _preview(files) -> None:
    st.subheader("Preview")
    columns = st.columns(4)
    for column, file in zip(columns, files[:4]):
        with column:
            try:
                file.seek(0)
                st.image(Image.open(file), width="stretch")
                st.caption(file.name[:28])
            except Exception:
                pass
