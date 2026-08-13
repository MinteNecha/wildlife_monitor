"""Settings page (UC8, FR9) — pipeline, thresholds, and hardware config."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from wildlife_monitor.config import SystemConfig
from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, rule, note

_DEFAULTS = {
    "pipeline": "bioclip_sam",
    "threshold": 0.50,
    "arch": "LSTM",
    "device": "cuda",
    "memory": 8,
    "top_n": 15,
    "activity": "diurnal, nocturnal, crepuscular",
    "movement": "migratory, territorial, nomadic",
    "social": "solitary, small group, large herd",
}


def render() -> None:
    header("System Configuration",
           "Pipeline, thresholds, model architecture and hardware · UC8")
    for key, value in _DEFAULTS.items():
        st.session_state.setdefault(key, value)

    st.caption("Values are validated before being applied. "
               "Invalid entries cannot be saved.")

    pending = _controls()
    errors = _validate(pending)
    rule()
    _apply_bar(pending, errors)
    rule()
    _active_table()


def _controls() -> dict:
    left, right = st.columns(2)
    pending: dict = {}
    pipelines = list(da.PIPELINE_DISPLAY.keys())

    with left:
        st.subheader("Detection Pipeline")
        pending["pipeline"] = st.selectbox(
            "Active pipeline", pipelines,
            index=pipelines.index(st.session_state["pipeline"]),
            format_func=lambda p: da.PIPELINE_DISPLAY[p]["label"])
        pending["threshold"] = st.slider(
            "Confidence threshold", 0.0, 1.0,
            float(st.session_state["threshold"]), 0.05)
        pending["top_n"] = st.number_input(
            "Images per run (top_n)", 1, 100, int(st.session_state["top_n"]), 1)
        st.subheader("Temporal Model (Pipeline 2 — planned)")
        pending["arch"] = st.radio(
            "Architecture", ["LSTM", "Transformer"],
            index=["LSTM", "Transformer"].index(st.session_state["arch"]),
            horizontal=True)

    with right:
        st.subheader("Hardware")
        pending["device"] = st.selectbox(
            "Device", ["cuda", "cpu"],
            index=["cuda", "cpu"].index(st.session_state["device"]))
        pending["memory"] = st.number_input(
            "Max memory (GB)", 2, 64, int(st.session_state["memory"]), 2)
        st.subheader("Behavioural Categories")
        pending["activity"] = st.text_input(
            "Activity timing", st.session_state["activity"])
        pending["movement"] = st.text_input(
            "Movement strategy", st.session_state["movement"])
        pending["social"] = st.text_input(
            "Social structure", st.session_state["social"])
    return pending


def _validate(pending: dict) -> list[tuple[str, str]]:
    return [(label, "needs at least two comma-separated categories")
            for label, key in [("Activity timing", "activity"),
                                ("Movement strategy", "movement"),
                                ("Social structure", "social")]
            if len([c for c in pending[key].split(",") if c.strip()]) < 2]


def _apply_bar(pending: dict, errors: list) -> None:
    changed = [k for k, v in pending.items() if st.session_state[k] != v]
    button_col, message_col = st.columns([1, 3])
    apply = button_col.button("Apply Configuration", disabled=bool(errors))
    with message_col:
        if errors:
            for label, message in errors:
                note(f"<b>{label}</b> — {message}", ok=False)
        elif changed:
            st.caption(f"Unsaved changes: {', '.join(changed)}")
        else:
            st.caption("No pending changes.")

    if apply and not errors:
        st.session_state.update(pending)
        _persist_to_system_config(pending)
        st.success("Configuration applied and saved to config.json.")


def _persist_to_system_config(pending: dict) -> None:
    """Write the runtime-relevant settings through the SystemConfig singleton."""
    config = SystemConfig.instance()
    config.device = pending["device"]
    config.confidence_threshold = float(pending["threshold"])
    config.top_n = int(pending["top_n"])
    config.save()


def _active_table() -> None:
    st.subheader("Active Configuration")
    st.dataframe(
        pd.DataFrame([{"Parameter": k, "Value": str(st.session_state[k])}
                      for k in _DEFAULTS]),
        width="stretch", height=300, hide_index=True)
    if st.button("Reset to Defaults"):
        st.session_state.update(_DEFAULTS)
        st.rerun()
