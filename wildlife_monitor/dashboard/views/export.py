"""Export page (UC7, FR8) — download detection data."""

from __future__ import annotations

import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule


def render(species: str) -> None:
    header("Export Results",
           "Download detection data for external analysis · UC7")

    detections = da.load_all_detections(species)
    if not detections:
        st.info(f"No detections for {da.pretty(species)} yet. Run a pipeline.")
        return

    _per_pipeline_downloads(species, detections)
    rule()
    _comparison_download(species)


def _per_pipeline_downloads(species, detections) -> None:
    st.subheader(f"Per-Pipeline Detections — {da.pretty(species)}")
    for pipeline, frame in detections.items():
        display = da.PIPELINE_DISPLAY.get(pipeline, {}).get("label", pipeline)
        accuracy = frame["correct"].mean() * 100
        with st.expander(f"{display} — {len(frame)} detections "
                         f"({accuracy:.1f}% accuracy)"):
            columns = st.columns(2)
            columns[0].metric("Rows", len(frame))
            columns[1].metric("Accuracy", f"{accuracy:.1f}%")
            st.dataframe(frame.head(8), width="stretch",
                         height=200, hide_index=True)
            st.download_button(
                f"Download {display} CSV",
                frame.to_csv(index=False).encode(),
                f"{pipeline}_{species}.csv", "text/csv", key=pipeline)


def _comparison_download(species: str) -> None:
    st.subheader("Comparison Report")
    report = da.load_comparison_report(species)
    if report:
        st.download_button("Download comparison report",
                           report.encode(),
                           f"comparison_report_{species}.txt", "text/plain")
        st.code(report, language="text")
    else:
        st.info(f"No comparison report. Run: "
                f"python scripts/run_compare.py --species {species}")
