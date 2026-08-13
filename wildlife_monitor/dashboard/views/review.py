"""Image Review page (UC2, UC6) — visual inspection of detections."""

from __future__ import annotations

import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import (
    header, image_count_slider, detection_grid,
)
from wildlife_monitor.dashboard.theme import BLACK, POS, NEG


def render(species: str, pipeline: str) -> None:
    header(f"Image Review — {da.pretty(species)}",
           "Visual inspection of detections and overlays · UC2, UC6")

    view = st.radio(
        "View mode",
        ["Detection Overlays", "Correct Detections", "Incorrect Detections"],
        horizontal=True, label_visibility="collapsed")

    if view == "Detection Overlays":
        _render_overlays(pipeline)
    else:
        _render_detections(species, pipeline, want_correct=view.startswith("Correct"))


def _render_overlays(pipeline: str) -> None:
    paths = da.overlay_paths(pipeline)
    if not paths:
        st.info(f"No overlays for {da.PIPELINE_DISPLAY.get(pipeline, {}).get('label', pipeline)}. "
                f"Run the pipeline to generate them.")
        return
    count = image_count_slider(len(paths))
    selected = paths[:count]
    for start in range(0, len(selected), 3):
        for column, path in zip(st.columns(3), selected[start:start + 3]):
            with column:
                st.image(Image.open(path), width="stretch")
                st.caption(path.stem[:38])


def _render_detections(species: str, pipeline: str, want_correct: bool) -> None:
    frame = da.load_detections(pipeline, species)
    if frame.empty:
        st.info("Run the pipeline first to review detections.")
        return

    subset = frame[frame["correct"] == want_correct].sort_values(
        "confidence", ascending=False)
    colour = POS if want_correct else NEG
    label = "correct" if want_correct else "incorrect"
    st.markdown(
        f"<div style='font-size:13px;color:{BLACK};margin-bottom:10px'>"
        f"<b style='color:{colour}'>{len(subset)}</b> {label} detections for "
        f"<b>{da.pretty(species)}</b></div>", unsafe_allow_html=True)

    if subset.empty:
        st.info(f"No {label} detections to display.")
        return
    count = image_count_slider(len(subset))
    if count:
        detection_grid(subset, count, per_row=3)
