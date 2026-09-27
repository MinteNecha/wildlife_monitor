"""Species Detection page (UC2) — per-species detection results."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import (
    header, rule, image_count_slider, detection_grid,
)


def render(species: str, pipeline: str) -> None:
    display = da.PIPELINE_DISPLAY.get(pipeline, {}).get("label", pipeline)
    header(f"Species Detection — {da.pretty(species)}",
           f"{display} detections, ranked by model confidence · UC2")

    frame = da.load_detections(pipeline, species)
    if frame.empty:
        st.info(f"No {display} results for {da.pretty(species)}. "
                f"Run: python scripts/run_pipeline.py "
                f"--pipeline {pipeline} --species {species}")
        return

    _metrics(frame)
    rule()
    filtered = _filter_controls(frame)
    if filtered.empty:
        st.info("No detections above this threshold. Lower the slider.")
        return

    _results_table(filtered)
    rule()
    st.subheader("Image Preview")
    count = image_count_slider(len(filtered), default=8, per_row=4)
    if count:
        detection_grid(filtered, count, per_row=4)


def _metrics(frame: pd.DataFrame) -> None:
    """Confidence-led metrics. Accuracy appears only where labels exist."""
    localised = int(frame["localised"].sum())
    needs_review = int((frame["band"] == "low").sum())
    accuracy = da.accuracy_of(frame)

    columns = st.columns(4)
    columns[0].metric("Detections", f"{len(frame):,}")
    columns[1].metric("Mean Confidence",
                      f"{frame['confidence'].mean() * 100:.1f}%")
    columns[2].metric("Needs Review", f"{needs_review:,}",
                      help="Detections below 50% confidence.")
    if accuracy is None:
        columns[3].metric("Accuracy", "—",
                          help="No ground-truth labels for these images, so "
                               "accuracy cannot be computed.")
    else:
        verification = da.verification_counts(frame)
        columns[3].metric(
            "Accuracy", f"{accuracy:.1f}%",
            help=f"Over the {verification['correct'] + verification['incorrect']:,} "
                 f"labelled detections only; "
                 f"{verification['unverified']:,} are unverified.")
    columns_note = ("Localised: "
                    f"{localised:,} of {len(frame):,} detections produced a "
                    f"bounding box or mask.")
    st.caption(columns_note)


def _filter_controls(frame: pd.DataFrame) -> pd.DataFrame:
    left, right = st.columns([2, 1])
    threshold = left.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.05)
    order = right.selectbox(
        "Sort by", ["Most confident first", "Least confident first"])

    filtered = frame[frame["confidence"] >= threshold]
    filtered = filtered.sort_values(
        "confidence", ascending=order.startswith("Least"))

    st.caption(f"Showing {len(filtered)} of {len(frame)} images at or above "
               f"{threshold:.0%} confidence.")
    return filtered


def _results_table(frame: pd.DataFrame) -> None:
    wanted = ["image_id", "camera_id", "species", "confidence", "band",
              "detection_quality", "location_type"]
    if da.has_ground_truth(frame):
        wanted.append("verification")

    columns = [column for column in wanted if column in frame.columns]
    display = frame[columns].assign(
        confidence=frame["confidence"].map("{:.1%}".format),
        band=frame["band"].map(da.band_label))
    st.dataframe(
        display, width="stretch", height=260, hide_index=True,
        column_config={
            "image_id": "Image",
            "camera_id": "Camera",
            "species": "Species (run for)",
            "confidence": "Confidence",
            "band": "Assessment",
            "detection_quality": "Quality",
            "location_type": "Output",
            "verification": "Against label"})
