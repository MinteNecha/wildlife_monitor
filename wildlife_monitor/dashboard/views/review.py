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
        _render_overlays(pipeline, species)
    else:
        _render_detections(species, pipeline, want_correct=view.startswith("Correct"))


def _render_overlays(pipeline: str, species: str) -> None:
    """Show overlays filtered to the selected species only."""
    all_paths = da.overlay_paths(pipeline)
    if not all_paths:
        st.info(f"No overlays for {da.PIPELINE_DISPLAY.get(pipeline, {}).get('label', pipeline)}. "
                f"Run the pipeline to generate them.")
        return

    # Filter overlay paths to those belonging to the selected species.
    # The detections CSV tells us exactly which image stems were processed
    # for this species — use those stems to filter the overlay folder.
    frame = da.load_detections(pipeline, species)
    if not frame.empty and "image_path" in frame.columns:
        import pathlib
        valid_stems = {
            pathlib.Path(str(p)).stem
            for p in frame["image_path"].dropna()
        }
        paths = [p for p in all_paths if p.stem.replace("_overlay", "") in valid_stems]
        if not paths:
            # Fallback: match by overlay stem containing species name
            paths = all_paths
    else:
        paths = all_paths

    if not paths:
        st.info(f"No overlays found for {da.pretty(species)}. Run the pipeline first.")
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
