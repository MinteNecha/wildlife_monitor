"""
Settings page (UC8, FR9) — the settings that take effect, and the constants
that do not.

An earlier version of this page offered nine controls. Six of them wrote to
session state and nowhere else: ``pipeline``, ``arch``, ``memory`` and the
three behavioural category lists. The page then reported "Configuration
applied and saved to config.json" for values no part of the system reads. A
control that cannot take effect is worse than no control, because someone who
changes it believes the system changed.

``SystemConfig`` holds three settings and the pipeline scripts read them.
Those stay editable and validated. Everything else is shown read only, taken
from the module that owns it rather than retyped here, with a note saying
where it is set instead.

The class lists are read only for a reason worth stating on the page: they are
baked into a trained model's output layer, and the thresholds produced the
labels the models were trained against. Editing either here would describe a
model that does not exist.
"""

from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from wildlife_monitor.config import SystemConfig, default_device
from wildlife_monitor.config.validator import (
    DEFAULTS as DECLARED_DEFAULTS, DEVICES, ConfigValidator,
)
from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.components import header, note, rule
from wildlife_monitor.pipeline2.aggregate import MIN_DETECTIONS
from wildlife_monitor.pipeline2.datasets import DEFAULT_EPOCHS
from wildlife_monitor.pipeline2.labelling import (
    ACTIVITY_CLASSES, MOVEMENT_CLASSES, SOCIAL_CLASSES, classify_movement,
)
from wildlife_monitor.pipeline2.sequence_builder import SequenceBuilder

_validator = ConfigValidator()

_FIELD_LABELS = {
    "device": "Device",
    "confidence_threshold": "Confidence threshold",
    "top_n": "Images per run",
}


def render(pipeline: str = "") -> None:
    header("System Configuration",
           "What this run uses, and where everything else is set · UC8")
    _seed_from_saved()

    st.caption("Three settings are stored in config.json and read by the "
               "pipeline scripts. They are validated before being saved, and "
               "an invalid value cannot be applied.")

    pending = _controls()
    errors = _validate(pending)
    rule()
    _apply_bar(pending, errors)
    rule()
    _fixed(pipeline)


# ── the three that persist ───────────────────────────────────────────────────

def _seed_from_saved() -> None:
    """Show what is actually saved, not a hardcoded guess at it."""
    config = SystemConfig.instance()
    st.session_state.setdefault("device", config.device)
    st.session_state.setdefault("confidence_threshold",
                                float(config.confidence_threshold))
    st.session_state.setdefault("top_n", int(config.top_n))


def _controls() -> dict:
    left, right = st.columns(2)
    devices = list(DEVICES)

    with left:
        st.subheader("Detection Run")
        threshold = st.slider(
            "Confidence threshold", 0.0, 1.0,
            float(st.session_state["confidence_threshold"]), 0.05,
            help=_validator.describe("confidence_threshold"))
        top_n = st.number_input(
            "Images per run", 1, 5000, int(st.session_state["top_n"]), 1,
            help=_validator.describe("top_n"))

    with right:
        st.subheader("Hardware")
        current = st.session_state["device"]
        device = st.selectbox(
            "Device", devices,
            index=devices.index(current) if current in devices else 0,
            help=_validator.describe("device"))
        st.caption(f"Detected on this machine: {default_device()}")

    return {"device": device, "confidence_threshold": threshold,
            "top_n": top_n}


def _validate(pending: dict) -> list[tuple[str, str]]:
    """Every editable field through ConfigValidator (FR9).

    UC8's alternative flow requires the valid range beside a rejected value,
    so the validator's own description is what is displayed.
    """
    return [(_FIELD_LABELS.get(failure.details.get("key", ""), "Setting"),
             failure.reason)
            for failure in _validator.validate_all(pending)]


def _apply_bar(pending: dict, errors: list) -> None:
    changed = [key for key, value in pending.items()
               if st.session_state[key] != value]
    button_column, message_column = st.columns([1, 3])
    apply = button_column.button("Apply Configuration", key="apply_config",
                                 disabled=bool(errors))
    with message_column:
        if errors:
            for label, message in errors:
                note(f"<b>{label}</b> {message}", ok=False)
        elif changed:
            st.caption("Unsaved changes: "
                       + ", ".join(_FIELD_LABELS[key] for key in changed))
        else:
            st.caption("No pending changes.")

    if apply and not errors:
        st.session_state.update(pending)
        _save(pending)

    if st.button("Reset to detected defaults", key="reset_config"):
        st.session_state.update(_declared_defaults())
        st.rerun()


def _save(pending: dict) -> None:
    config = SystemConfig.instance()
    config.device = pending["device"]
    config.confidence_threshold = float(pending["confidence_threshold"])
    config.top_n = int(pending["top_n"])
    config.save()
    st.success(f"Saved to config.json · device {config.device} · threshold "
               f"{config.confidence_threshold:.2f} · {config.top_n} images "
               f"per run.")


def _declared_defaults() -> dict:
    return {"device": default_device(),
            "confidence_threshold": float(DECLARED_DEFAULTS["threshold"]),
            "top_n": int(DECLARED_DEFAULTS["top_n"])}


# ── everything else, stated rather than offered ──────────────────────────────

def _fixed(pipeline: str) -> None:
    st.subheader("Fixed for This Build")
    st.caption("These are not editable here. The class lists are part of a "
               "trained model's output layer, and the thresholds produced the "
               "labels the models were trained against, so changing them on "
               "this page would describe a model that does not exist. The "
               "third column says where each one is set.")
    st.dataframe(pd.DataFrame(_fixed_rows(pipeline)),
                 width="stretch", height=420, hide_index=True)


def _fixed_rows(pipeline: str) -> list[dict[str, str]]:
    rule_defaults = _rule_defaults()
    architectures = ", ".join(_arch(name) for name in DEFAULT_EPOCHS)
    epochs = ", ".join(f"{_arch(name)} {count}"
                       for name, count in DEFAULT_EPOCHS.items())
    return [
        _row("Active pipeline",
             da.PIPELINE_DISPLAY[pipeline]["label"] if pipeline in
             da.PIPELINE_DISPLAY else "none selected",
             "Pipeline selector in the sidebar"),
        _row("Temporal architecture", architectures,
             "Chosen per view on the Behavioural Analysis page"),
        _row("Training epochs", epochs, "pipeline2/datasets.py"),
        _row("Sequence cap",
             f"{SequenceBuilder().max_length} detections per camera",
             "pipeline2/sequence_builder.py"),
        _row("Activity classes", ", ".join(ACTIVITY_CLASSES),
             "pipeline2/labelling.py"),
        _row("Movement classes", ", ".join(MOVEMENT_CLASSES),
             "pipeline2/labelling.py"),
        _row("Group size classes", ", ".join(SOCIAL_CLASSES),
             "pipeline2/labelling.py"),
        _row("Territorial cameras",
             _percentile(rule_defaults["territorial_percentile"])
             + " by site fidelity",
             "classify_movement()"),
        _row("Migratory cameras",
             _percentile(rule_defaults["migratory_percentile"])
             + " by busiest-month share",
             "classify_movement()"),
        _row("Detections needed for a migratory call",
             str(rule_defaults["min_detections_for_migratory"]),
             "classify_movement()"),
        _row("Detections needed before a camera votes", str(MIN_DETECTIONS),
             "pipeline2/aggregate.py"),
    ]


def _row(setting: str, value: str, source: str) -> dict[str, str]:
    return {"Setting": setting, "Value": value, "Set by": source}


def _rule_defaults() -> dict:
    """The movement thresholds, read from the function that applies them."""
    return {name: parameter.default for name, parameter
            in inspect.signature(classify_movement).parameters.items()}


def _percentile(value: float) -> str:
    return f"top {round((1.0 - float(value)) * 100)}% of cameras"


def _arch(name: str) -> str:
    return "LSTM" if name == "lstm" else name.title()