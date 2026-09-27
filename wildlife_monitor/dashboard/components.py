"""
Reusable dashboard UI components.

Small presentational helpers shared across pages: section headers, rules,
result notes, and image grids. Keeping them here removes repetition from
the page modules and keeps their markup consistent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.theme import BLACK, GREY, POS, NEG

# Colour per confidence band. Low confidence is flagged not because it is
# wrong — without ground truth that is unknowable — but because it is where a
# human reviewer's attention is worth spending.
BAND_COLOURS = {"high": POS, "medium": "#E67E22", "low": NEG}


def header(title: str, subtitle: str = "") -> None:
    """Render a page header with an optional subtitle."""
    st.markdown(
        f"<div class='hd'><div class='hd-title'>{title}</div>"
        f"<div class='hd-sub'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )


def rule() -> None:
    """Render a horizontal rule."""
    st.markdown("<div class='hrule'></div>", unsafe_allow_html=True)


def note(text: str, ok: bool = True) -> None:
    """Render a coloured result note (green for ok, red for problems)."""
    colour = POS if ok else NEG
    background = "#D1FAE5" if ok else "#FEE2E2"
    st.markdown(
        f"<div style='background:{background};border-left:3px solid {colour};"
        f"border-radius:3px;padding:8px 10px;margin-bottom:6px;"
        f"font-size:12px;color:{BLACK}'>{text}</div>",
        unsafe_allow_html=True,
    )


def image_count_slider(total: int, default: int = 9, per_row: int = 3) -> int:
    """Return how many images to show, with a slider when there are many."""
    if total <= per_row + 1:
        return max(total, 0)
    high = min(24, total)
    return st.slider("Images to display", per_row, high, min(default, high))


def detection_grid(frame: pd.DataFrame, count: int, per_row: int = 4) -> None:
    """Render a grid of detection thumbnails with verdict captions."""
    rows = [row for _, row in frame.head(count).iterrows()]
    for start in range(0, len(rows), per_row):
        columns = st.columns(per_row)
        for column, row in zip(columns, rows[start:start + per_row]):
            with column:
                _detection_tile(row)


def _detection_tile(row: pd.Series) -> None:
    """Render one detection thumbnail with its verdict and metadata."""
    image_path = Path(str(row.get("image_path", "")))
    if image_path.exists():
        st.image(Image.open(image_path), width="stretch")

    correct = bool(row.get("correct", False))
    colour = POS if correct else NEG
    verdict = "Correct" if correct else "Incorrect"
    species = da.pretty(row.get("species", "?"))
    confidence = float(row.get("confidence", 0.0))
    site = row.get("camera_id", "?")

    st.markdown(
        f"<div style='font-size:11px;font-weight:700;color:{colour}'>{verdict}</div>"
        f"<div style='font-size:11px;color:{BLACK}'>{species} · {confidence:.0%}</div>"
        f"<div style='font-size:10px;color:{GREY}'>Site {site}</div>",
        unsafe_allow_html=True,
    )


STATUS_COLOURS = {"supported": POS, "limited": "#E67E22", "unsupported": NEG}
_CAPABILITY_TITLES = {
    "activity": "Activity timing (day / night)",
    "movement": "Movement strategy (migratory / territorial / nomadic)",
    "social": "Social structure (group size)",
}


def sufficiency_panel(report, show_guidance: bool = True) -> None:
    """Render what the available data can and cannot honestly support.

    The behavioural rules never fail loudly on thin data — they compare each
    camera against the others and always return a confident-looking label. This
    panel is what stops the dashboard presenting that as fact.
    """
    st.markdown(
        f"<div style='font-size:12px;color:{GREY};margin-bottom:8px'>"
        f"{report.summary_line()}</div>", unsafe_allow_html=True)

    for capability in report.capabilities:
        colour = STATUS_COLOURS.get(capability.status, GREY)
        background = {"supported": "#D1FAE5", "limited": "#FEF3C7",
                      "unsupported": "#FEE2E2"}.get(capability.status, "#F3F4F6")
        title = _CAPABILITY_TITLES.get(capability.name,
                                        capability.name.title())
        st.markdown(
            f"<div style='background:{background};border-left:3px solid {colour};"
            f"border-radius:3px;padding:8px 10px;margin-bottom:6px'>"
            f"<div style='font-size:12px;font-weight:700;color:{BLACK}'>"
            f"{title} — <span style='color:{colour}'>{capability.label}</span></div>"
            f"<div style='font-size:11px;color:{GREY};margin-top:3px'>"
            f"{capability.reason}</div></div>", unsafe_allow_html=True)

    if show_guidance:
        from wildlife_monitor.data.sufficiency import guidance_for
        guidance = guidance_for(report)
        if guidance:
            st.caption(guidance)
