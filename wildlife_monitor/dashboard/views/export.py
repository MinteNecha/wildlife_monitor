"""Export page (UC7, FR8) — download detection and behaviour results."""

from __future__ import annotations

import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule
from wildlife_monitor.data.export import ExportService

_exporter = ExportService()
FORMATS = ["CSV", "JSON", "PDF report"]


def render(species: str) -> None:
    header("Export Results",
           "Download detection and behaviour data for external analysis · UC7")

    detections = da.load_all_detections(species)
    behaviour = _behaviour_frame(species)
    if not detections and behaviour.empty:
        st.info(f"No results for {da.pretty(species)} yet. Run a pipeline.")
        return

    export_format = st.radio("Format", FORMATS, horizontal=True)

    if detections:
        _per_pipeline_downloads(species, detections, export_format)
    if not behaviour.empty:
        rule()
        _behaviour_download(species, behaviour, export_format)
    rule()
    _comparison_download(species)


def _behaviour_frame(species: str):
    """Behaviour predictions for whichever architecture is trained."""
    architectures = da.behaviour_architectures(species)
    if not architectures:
        import pandas as pd
        return pd.DataFrame()
    preferred = str(st.session_state.get("arch", "lstm")).lower()
    architecture = preferred if preferred in architectures else architectures[0]
    service = da.load_behaviour_service(species, architecture)
    if service is None:
        import pandas as pd
        return pd.DataFrame()
    return service.predict_frame(da.behaviour_detections(species), species)


def _payload(frame, export_format: str, title: str, summary, columns):
    """Serialise a frame into the selected format."""
    if export_format == "CSV":
        return _exporter.to_csv(frame), "csv", "text/csv"
    if export_format == "JSON":
        return _exporter.to_json(frame), "json", "application/json"
    return (_exporter.to_pdf(frame, title=title, summary=summary,
                             columns=columns),
            "pdf", "application/pdf")


def _per_pipeline_downloads(species, detections, export_format) -> None:
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

            payload, extension, mime = _payload(
                frame, export_format,
                title=f"{da.pretty(species)} — {display}",
                summary=[("Species", da.pretty(species)),
                         ("Pipeline", display),
                         ("Detections", len(frame)),
                         ("Accuracy", f"{accuracy:.1f}%")],
                columns=["image_id", "camera_id", "timestamp",
                         "confidence", "instance_count"])
            st.download_button(
                f"Download {display} ({export_format})", payload,
                f"{pipeline}_{species}.{extension}", mime,
                key=f"{pipeline}_{extension}")


def _behaviour_download(species, behaviour, export_format) -> None:
    st.subheader("Behavioural Classifications")
    st.caption("Per-camera activity, movement and social structure from "
               "Pipeline 2, with the model's confidence in each prediction.")
    st.dataframe(behaviour.head(8), width="stretch", height=200,
                 hide_index=True)

    counts = behaviour["movement_class"].value_counts().to_dict()
    payload, extension, mime = _payload(
        behaviour, export_format,
        title=f"{da.pretty(species)} — Behavioural Analysis",
        summary=[("Species", da.pretty(species)),
                 ("Cameras classified", len(behaviour)),
                 *[(f"Movement: {name}", count)
                   for name, count in counts.items()]],
        columns=["camera_id", "activity_class", "movement_class",
                 "social_class", "detection_count"])
    st.download_button(
        f"Download behaviour results ({export_format})", payload,
        f"behaviour_{species}.{extension}", mime, key=f"behaviour_{extension}")


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
