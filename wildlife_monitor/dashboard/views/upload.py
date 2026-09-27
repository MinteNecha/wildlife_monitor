"""Upload page (UC1, FR1) — batch ingestion with validation."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard.components import header, rule, note
from wildlife_monitor.dashboard.theme import BLACK, GREY
from wildlife_monitor.data.sufficiency import MIN_SPAN_MONTHS
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
    _coverage_summary(accepted)
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


def _coverage_summary(accepted) -> None:
    """Date range covered by this batch, and what that span can support.

    Time span is the constraint most likely to invalidate a result, and it is
    knowable at upload time from EXIF alone — long before any detection has
    run. Saying so here is cheaper than letting someone process a fortnight of
    photographs and then discover the movement classes were meaningless.
    """
    st.subheader("Coverage")
    stamps = sorted(
        result.details.get("timestamp", "") for result in accepted
        if result.details.get("has_timestamp"))

    if not stamps:
        note("None of these images carry an EXIF capture time. Without "
             "timestamps the system cannot order detections, so no "
             "behavioural analysis is possible. Supply a metadata CSV with "
             "capture times.", ok=False)
        return

    first, last = stamps[0][:10], stamps[-1][:10]
    span_days = (pd.to_datetime(last) - pd.to_datetime(first)).days
    span_months = span_days / 30.44

    columns = st.columns(3)
    columns[0].metric("Earliest", first)
    columns[1].metric("Latest", last)
    columns[2].metric("Span", f"{span_months:.1f} months")

    if span_months >= MIN_SPAN_MONTHS[1]:
        note(f"This batch spans {span_months:.1f} months — enough coverage for "
             f"seasonal movement analysis once detections have run.", ok=True)
    elif span_months >= MIN_SPAN_MONTHS[0]:
        note(f"This batch spans {span_months:.1f} months. Activity timing will "
             f"be reliable. Movement classification will work but should be "
             f"read as indicative — {MIN_SPAN_MONTHS[1]} months or more makes "
             f"it dependable.", ok=False)
    else:
        note(f"This batch spans {span_months:.1f} months. Activity timing "
             f"(day/night) will work. Movement classification will not: "
             f"migratory and territorial differ by how presence is spread "
             f"across seasons, and under {MIN_SPAN_MONTHS[0]} months every "
             f"camera looks alike. More photographs over the same window will "
             f"not change this — more months will.", ok=False)

    st.markdown(
        f"<div style='font-size:11px;color:{GREY};margin-top:4px'>"
        f"Camera sites are read from the folder each image sits in at "
        f"ingestion, so they are not shown for a browser upload.</div>",
        unsafe_allow_html=True)


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
