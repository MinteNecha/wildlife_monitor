"""
Image Review page (UC2, UC6) — visual inspection of detections.

Review is organised by model confidence rather than by correctness. That is a
deliberate change from the earlier version, which split detections into
"correct" and "incorrect" grids.

Those grids only worked because Snapshot Serengeti ships with citizen-science
labels for every image. A user with their own camera trap photographs has no
labels — that is the reason they are running a classifier at all — so the
split would be empty or, worse, would report everything as incorrect.

Confidence is available for every detection whether or not a label exists, and
it points a reviewer at the right images: you cannot know which predictions
are wrong without ground truth, but you always know which ones the model was
unsure about. Reviewing those is how ground truth gets created.
"""

from __future__ import annotations

import pathlib

import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import (
    header, rule, image_count_slider, detection_grid, BAND_COLOURS,
)
from wildlife_monitor.dashboard.theme import BLACK, GREY

_BAND_ORDER = ["low", "medium", "high"]



def render(species: str, pipeline: str) -> None:
    header(f"Image Review — {da.pretty(species)}",
           "Visual inspection, ordered by what needs attention · UC2, UC6")

    view = st.radio("View mode", ["Detections", "Detection Overlays"],
                    horizontal=True, label_visibility="collapsed")
    if view == "Detection Overlays":
        _render_overlays(pipeline, species)
    else:
        _render_detections(species, pipeline)


def _render_detections(species: str, pipeline: str) -> None:
    frame = da.load_detections(pipeline, species)
    if frame.empty:
        st.info("Run the pipeline first to review detections.")
        return

    _band_summary(frame)
    rule()

    order = st.selectbox("Order", ["Least confident first",
                                    "Most confident first"])
    subset = frame.sort_values("confidence",
                               ascending=order.startswith("Least"))
    st.caption("Least confident first puts the detections the model was "
               "unsure about at the top. Those are the ones worth checking, "
               "and reviewing them is how ground truth gets created.")


    st.markdown(
        f"<div style='font-size:13px;color:{BLACK};margin-bottom:10px'>"
        f"<b>{len(subset)}</b> detections in view for "
        f"<b>{da.pretty(species)}</b></div>", unsafe_allow_html=True)

    if subset.empty:
        st.info("No detections in the selected bands.")
        return

    count = image_count_slider(len(subset))
    if count:
        detection_grid(subset, count, per_row=3)


def _band_summary(frame) -> None:
    """Confidence split, plus verification only where labels actually exist."""
    counts = frame["band"].value_counts()
    columns = st.columns(4)
    columns[0].metric("Detections", f"{len(frame):,}")
    for column, band in zip(columns[1:], _BAND_ORDER):
        value = int(counts.get(band, 0))
        share = value / len(frame) * 100 if len(frame) else 0
        column.metric(da.band_label(band), f"{value:,}", f"{share:.0f}%",
                      delta_color="off")

    if da.has_ground_truth(frame):
        verification = da.verification_counts(frame)
        accuracy = da.accuracy_of(frame)
        st.caption(
            f"Ground-truth labels are available for "
            f"{verification['correct'] + verification['incorrect']:,} of "
            f"{len(frame):,} detections, of which "
            f"{accuracy:.1f}% match the predicted species. The remaining "
            f"{verification['unverified']:,} are unverified — not wrong, "
            f"simply unlabelled.")
    else:
        st.caption("These images carry no ground-truth labels, so accuracy "
                   "cannot be computed. Confidence is the available signal, "
                   "and your review is what turns it into ground truth.")


def _render_overlays(pipeline: str, species: str) -> None:
    """Show overlays filtered to the selected species only."""
    all_paths = da.overlay_paths(pipeline)
    if not all_paths:
        label = da.PIPELINE_DISPLAY.get(pipeline, {}).get("label", pipeline)
        st.info(f"No overlays for {label}. Run the pipeline to generate them.")
        return

    frame = da.load_detections(pipeline, species)
    paths = all_paths
    if not frame.empty and "image_path" in frame.columns:
        stems = {pathlib.Path(str(path)).stem
                 for path in frame["image_path"].dropna()}
        matched = [path for path in all_paths
                   if path.stem.replace("_overlay", "") in stems]
        paths = matched or all_paths

    count = image_count_slider(len(paths))
    selected = paths[:count]
    for start in range(0, len(selected), 3):
        for column, path in zip(st.columns(3), selected[start:start + 3]):
            with column:
                st.image(Image.open(path), width="stretch")
                st.caption(path.stem[:38])
