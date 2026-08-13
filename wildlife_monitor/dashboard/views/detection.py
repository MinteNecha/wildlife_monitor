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
           f"{display} detections against citizen-scientist labels · UC2")

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
    accuracy = frame["correct"].mean() * 100
    localised = int(frame["localised"].sum())
    columns = st.columns(4)
    columns[0].metric("Images Evaluated", len(frame))
    columns[1].metric("Accuracy", f"{accuracy:.1f}%")
    columns[2].metric("Mean Confidence",
                      f"{frame['confidence'].mean() * 100:.1f}%")
    columns[3].metric("Localised", f"{localised} / {len(frame)}")


def _filter_controls(frame: pd.DataFrame) -> pd.DataFrame:
    left, right = st.columns([2, 1])
    threshold = left.slider("Confidence threshold", 0.0, 1.0, 0.0, 0.05)
    order = right.selectbox(
        "Sort by", ["Confidence (high)", "Confidence (low)", "Correct first"])

    filtered = frame[frame["confidence"] >= threshold]
    if order.startswith("Confidence"):
        filtered = filtered.sort_values(
            "confidence", ascending=order == "Confidence (low)")
    else:
        filtered = filtered.sort_values("correct", ascending=False)

    st.caption(f"Showing {len(filtered)} of {len(frame)} images at or above "
               f"{threshold:.0%} confidence.")
    return filtered


def _results_table(frame: pd.DataFrame) -> None:
    columns = [c for c in ["image_id", "species", "confidence",
                           "detection_quality", "location_type", "correct"]
               if c in frame.columns]
    display = frame[columns].assign(
        confidence=frame["confidence"].map("{:.1%}".format))
    st.dataframe(
        display, width="stretch", height=260, hide_index=True,
        column_config={
            "species": "Species (run for)",
            "confidence": "Confidence",
            "detection_quality": "Quality",
            "location_type": "Output",
            "correct": st.column_config.CheckboxColumn("Correct")})
