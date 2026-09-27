"""Overview page (UC2, UC3) — dataset summary and accuracy charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule
from wildlife_monitor.dashboard.theme import (
    PLOT, GRID, CHART_SEQ, CHART_RAMP, POS_BAR, NEG_BAR, WHITE, BLACK,
)


def render(species: str) -> None:
    header("Dataset Overview",
           "Snapshot Serengeti Season 1 · pipeline detections · UC2, UC3")

    subset = da.load_subset()
    if subset.empty:
        st.info("No subset metadata found. Run 'python scripts/setup_data.py'.")
        return

    all_detections = da.load_all_detections(species)
    combined = (pd.concat(all_detections.values(), ignore_index=True)
                if all_detections else pd.DataFrame())

    _metric_row(species, subset, combined, all_detections)
    rule()
    _accuracy_and_distribution(subset, combined)
    if not combined.empty:
        rule()
        _confidence_distribution(combined)


def _metric_row(species, subset, combined, all_detections) -> None:
    sites = (subset["site_id"].nunique()
             if "site_id" in subset.columns else "—")
    accuracy = da.accuracy_of(combined) if not combined.empty else None
    confidence = (f"{combined['confidence'].mean() * 100:.1f}%"
                  if not combined.empty else "—")

    columns = st.columns(5)
    for column, (label, value) in zip(columns, [
        ("Total Images", f"{len(subset):,}"),
        ("Species", subset["species_label"].nunique()),
        (f"Mean Confidence ({da.pretty(species)})", confidence),
        ("Camera Sites", sites),
        ("Pipelines Run", len(all_detections)),
    ]):
        column.metric(label, value)

    # Accuracy is only meaningful against ground-truth labels, which most
    # deployments will not have. Report it as a note where it exists rather
    # than as a headline metric that reads as zero when it does not.
    if accuracy is not None:
        verification = da.verification_counts(combined)
        labelled = verification["correct"] + verification["incorrect"]
        st.caption(f"Against citizen-science labels: {accuracy:.1f}% of the "
                   f"{labelled:,} labelled detections match the predicted "
                   f"species. {verification['unverified']:,} detections are "
                   f"unverified.")
    elif not combined.empty:
        st.caption("These detections carry no ground-truth labels, so accuracy "
                   "cannot be computed. Confidence is the available signal.")


def _accuracy_and_distribution(subset, combined) -> None:
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Detection Confidence by Pipeline")
        if combined.empty:
            st.info("Run a pipeline to see results here.")
        else:
            summary = (combined.groupby("pipeline")["confidence"]
                       .mean().mul(100).reset_index(name="conf"))
            summary["label"] = summary["pipeline"].map(
                lambda p: da.PIPELINE_DISPLAY.get(p, {}).get("label", p))
            figure = go.Figure(go.Bar(
                x=summary["conf"], y=summary["label"], orientation="h",
                marker=dict(color=summary["conf"], colorscale=CHART_RAMP,
                            cmin=0, cmax=100, line=dict(width=0)),
                text=summary["conf"].map("{:.1f}%".format),
                textposition="outside", textfont=dict(color=BLACK, size=11)))
            figure.update_layout(
                **PLOT, height=300, bargap=0.4,
                xaxis=dict(range=[0, 118], title="Mean Confidence (%)", **GRID),
                yaxis=dict(showgrid=False, tickfont=dict(color=BLACK)))
            st.plotly_chart(figure, width="stretch")

    with right:
        st.subheader("Images per Species")
        distribution = subset["species_label"].value_counts().reset_index()
        distribution.columns = ["species", "n"]
        distribution["species"] = distribution["species"].map(da.pretty)
        figure = px.pie(distribution, values="n", names="species", hole=0.5,
                        color_discrete_sequence=CHART_SEQ)
        figure.update_traces(
            textinfo="percent", textfont=dict(size=11, color=WHITE),
            hovertemplate="<b>%{label}</b><br>%{value} images<extra></extra>")
        figure.update_layout(**PLOT, height=300,
                             legend=dict(font=dict(size=10, color=BLACK),
                                         bgcolor=WHITE))
        st.plotly_chart(figure, width="stretch")


def _confidence_distribution(combined) -> None:
    """Confidence spread, split by verification only where labels exist."""
    st.subheader("Confidence Score Distribution")
    figure = go.Figure()

    if da.has_ground_truth(combined):
        for state, label, colour in [("correct", "Matches label", POS_BAR),
                                      ("incorrect", "Differs from label", NEG_BAR)]:
            values = combined.loc[combined["verification"] == state, "confidence"]
            if not values.empty:
                figure.add_trace(go.Histogram(
                    x=values, name=label, nbinsx=25,
                    marker_color=colour, opacity=0.80))
        caption = ("Where the two overlap, the model is equally confident "
                   "whether or not it agrees with the label — a sign of "
                   "overconfidence.")
    else:
        figure.add_trace(go.Histogram(
            x=combined["confidence"], name="Detections", nbinsx=25,
            marker_color=POS_BAR, opacity=0.85))
        caption = ("A long left tail means many low-confidence detections, "
                   "which are the ones worth reviewing by eye.")

    figure.update_layout(
        **PLOT, height=220, barmode="overlay",
        legend=dict(font=dict(size=11, color=BLACK), bgcolor=WHITE),
        xaxis=dict(title="Confidence Score", **GRID),
        yaxis=dict(title="Count", **GRID))
    st.plotly_chart(figure, width="stretch")
    st.caption(caption)
