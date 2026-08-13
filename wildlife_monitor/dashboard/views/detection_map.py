"""Detection Map page (UC5) — camera sites across the Serengeti."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule
from wildlife_monitor.dashboard.theme import PLOT, POS_BAR, NEG_BAR


def render(species: str, pipeline: str) -> None:
    header(f"Detection Map — {da.pretty(species)}",
           "Camera sites across Serengeti National Park · UC5")

    frame = da.load_detections(pipeline, species)
    if frame.empty or "latitude" not in frame.columns:
        st.info("No geolocated detections. Ensure the subset metadata has "
                "latitude and longitude, and run a pipeline.")
        return

    frame = frame.dropna(subset=["latitude", "longitude"])
    frame = frame[(frame["latitude"] != 0) | (frame["longitude"] != 0)]
    if frame.empty:
        st.info("Detections have no valid coordinates.")
        return

    _render_map(species, frame)
    rule()
    _site_summary(frame)


def _render_map(species: str, frame: pd.DataFrame) -> None:
    try:
        import folium
        from streamlit_folium import st_folium

        canvas = folium.Map([-2.33, 34.83], zoom_start=10,
                            tiles="CartoDB positron")
        for _, row in frame.iterrows():
            correct = bool(row.get("correct", False))
            folium.CircleMarker(
                [row["latitude"], row["longitude"]],
                radius=9, color="white", weight=2, fill=True, fill_opacity=0.85,
                fill_color=POS_BAR if correct else NEG_BAR,
                tooltip=f"{row.get('camera_id', '?')} · "
                        f"{float(row['confidence']):.0%}",
                popup=folium.Popup(
                    f"<div style='font-size:12px;font-family:sans-serif'>"
                    f"<b>{da.pretty(species)}</b><br>"
                    f"Site: {row.get('camera_id', '?')}<br>"
                    f"Habitat: {row.get('habitat_type', '?')}<br>"
                    f"Confidence: {float(row['confidence']):.1%}<br>"
                    f"<b style='color:{'green' if correct else 'red'}'>"
                    f"{'Correct' if correct else 'Incorrect'}</b></div>",
                    max_width=220)).add_to(canvas)
        st_folium(canvas, height=470, returned_objects=[])
    except ImportError:
        figure = px.scatter_mapbox(
            frame, lat="latitude", lon="longitude", color="correct", zoom=8.5,
            mapbox_style="carto-positron",
            color_discrete_map={True: POS_BAR, False: NEG_BAR},
            hover_data=["camera_id", "confidence"])
        figure.update_layout(**PLOT, height=470)
        st.plotly_chart(figure, width="stretch")


def _site_summary(frame: pd.DataFrame) -> None:
    st.subheader("Summary by Camera Site")
    keys = [k for k in ["camera_id", "habitat_type"] if k in frame.columns]
    summary = (frame.groupby(keys)
               .agg(images=("image_id", "count"),
                    correct=("correct", "sum"),
                    avg_conf=("confidence", "mean"))
               .reset_index().sort_values("images", ascending=False))
    summary["accuracy"] = (summary["correct"] / summary["images"] * 100
                           ).map("{:.0f}%".format)
    summary["avg_conf"] = summary["avg_conf"].map("{:.1%}".format)
    st.dataframe(summary, width="stretch", height=240, hide_index=True)
