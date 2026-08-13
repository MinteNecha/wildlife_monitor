"""
Dashboard theme: palette, CSS, and Plotly styling.

Isolating all presentation constants here keeps the page modules focused
on content rather than colours and markup. ``inject_css`` is called once
at startup; ``PLOT`` and ``GRID`` are spread into Plotly layouts.
"""

from __future__ import annotations

import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────
ACCENT = "#2D6A4F"
LIGHT_BG = "#F7F8FA"
WHITE = "#FFFFFF"
CARD_BORDER = "#D0D5D2"
BLACK = "#111111"
GREY = "#444444"
POS = "#1B7A4A"
NEG = "#B33A2A"
POS_BAR = "#2D9E5F"
NEG_BAR = "#E05C3A"

CHART_SEQ = ["#2D6A4F", "#3A8C6A", "#E67E22", "#2980B9",
             "#8E44AD", "#C0392B", "#16A085", "#F39C12"]
CHART_RAMP = [[0, "#E05C3A"], [0.5, "#F4D03F"], [1, "#2D9E5F"]]
PIPELINE_COLOURS = ["#2D6A4F", "#2980B9", "#8E44AD"]

# ── Plotly layout defaults ────────────────────────────────────────────────────
PLOT = dict(
    paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(family="Arial, sans-serif", color=BLACK, size=12),
    margin=dict(l=0, r=10, t=14, b=0),
)
GRID = dict(
    showgrid=True, gridcolor="#E5E9E6", zeroline=False,
    tickfont=dict(color=BLACK), title_font=dict(color=BLACK),
)


def inject_css() -> None:
    """Pin a light theme with black text on every Streamlit surface."""
    st.markdown(f"""
    <style>
      .stApp, .main, section[data-testid="stMain"],
      [data-testid="stMainBlockContainer"],
      .block-container, [data-testid="stVerticalBlock"] {{
          background-color: {LIGHT_BG} !important;
          color: {BLACK} !important;
      }}
      [data-testid="stSidebar"], [data-testid="stSidebar"] > div,
      [data-testid="stSidebarNav"] {{
          background-color: {WHITE} !important;
          border-right: 1px solid {CARD_BORDER} !important;
      }}
      [data-testid="stSidebar"] * {{ color: {BLACK} !important; }}
      .stApp p, .stApp span, .stApp div, .stApp label,
      .stApp li, .stApp a, .stApp small, .stApp caption,
      .stMarkdown, .stMarkdown *, .stText, .stText *,
      [data-testid="stWidgetLabel"] *, [data-testid="stCaptionContainer"] *,
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
      [data-testid="stMetricDelta"] {{ color: {BLACK} !important; }}
      h1, h2, h3, h4, h5, h6 {{ color: {BLACK} !important; font-weight: 700 !important; }}
      h2 {{ font-size: 16px !important; }}
      input, textarea, select {{
          background-color: {WHITE} !important; color: {BLACK} !important;
      }}
      .hd {{ margin-bottom: 6px; }}
      .hd-title {{ font-size: 22px; font-weight: 700; color: {BLACK}; }}
      .hd-sub {{ font-size: 12px; color: {GREY}; margin-top: 2px; }}
      .hrule {{ height: 1px; background: {CARD_BORDER};
                margin: 14px 0 16px; border: none; }}
    </style>
    """, unsafe_allow_html=True)
