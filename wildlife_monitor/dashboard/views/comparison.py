"""Pipeline Comparison page (UC3) — three pipelines side by side."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule
from wildlife_monitor.dashboard.theme import PLOT, GRID, PIPELINE_COLOURS, GREY, BLACK


def render(species: str) -> None:
    header("Pipeline Comparison",
           "BioCLIP+SAM 3 vs BioCLIP+YOLO vs BioCLIP+MegaDetector · UC3")

    detections = da.load_all_detections(species)
    if not detections:
        st.info(f"No pipeline results for {da.pretty(species)} yet.\n\n"
                f"Run: python scripts/run_compare.py --species {species}")
        return

    _pipeline_cards(detections)
    rule()
    _quality_and_report(species, detections)


def _pipeline_cards(detections: dict[str, pd.DataFrame]) -> None:
    columns = st.columns(len(detections))
    for column, colour, (pipeline, frame) in zip(
            columns, PIPELINE_COLOURS, detections.items()):
        display = da.PIPELINE_DISPLAY.get(pipeline, {})
        localised = int(frame["localised"].sum()) if "localised" in frame else 0
        quality = (frame["detection_quality"].mean()
                   if "detection_quality" in frame else 0.0)
        with column.container(border=True):
            st.markdown(
                f"<div style='font-size:14px;font-weight:700;color:{colour}'>"
                f"{display.get('label', pipeline)}</div>"
                f"<div style='font-size:11px;color:{GREY};margin-bottom:10px'>"
                f"{display.get('output', '')}</div>", unsafe_allow_html=True)
            metrics = st.columns(3)
            metrics[0].metric("Processed", len(frame))
            metrics[1].metric("Localised", localised)
            metrics[2].metric("Quality", f"{quality:.3f}")


def _quality_and_report(species, detections) -> None:
    left, right = st.columns(2)

    with left:
        st.subheader("Detection Quality by Pipeline")
        rows = [{"Pipeline": da.PIPELINE_DISPLAY.get(p, {}).get("label", p),
                 "Quality": value}
                for p, frame in detections.items()
                if "detection_quality" in frame
                for value in frame["detection_quality"].dropna()]
        if rows:
            figure = px.box(pd.DataFrame(rows), x="Pipeline", y="Quality",
                            points="all", color="Pipeline",
                            color_discrete_sequence=PIPELINE_COLOURS)
            figure.update_traces(marker=dict(size=5, opacity=0.65))
            figure.update_layout(**PLOT, height=320, showlegend=False,
                                yaxis=dict(title="Quality Score", **GRID),
                                xaxis=dict(showgrid=False,
                                           tickfont=dict(color=BLACK)))
            st.plotly_chart(figure, width="stretch")
        else:
            st.info("No quality scores to plot yet.")

    with right:
        st.subheader("Comparison Report")
        report = da.load_comparison_report(species)
        if report:
            # Use a styled div so the report is always readable
            # regardless of Streamlit theme (st.code uses dark bg).
            escaped = report.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
            st.markdown(
                f"<div style='"
                f"background:#FFFFFF;color:#111111;font-family:monospace;"
                f"font-size:11px;line-height:1.5;padding:12px;"
                f"border:1px solid #D0D5D2;border-radius:4px;"
                f"white-space:pre;overflow-x:auto;max-height:420px;"
                f"overflow-y:auto'>{escaped}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info(f"No comparison report yet.\n\nRun: "
                    f"python scripts/run_compare.py --species {species}")
