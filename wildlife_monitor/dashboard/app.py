"""
Wildlife Monitor dashboard — Streamlit entry point.

Run from the project root:
    streamlit run wildlife_monitor/dashboard/app.py

The app is a thin shell: it configures the page, renders the sidebar
(navigation + species/pipeline selectors), and routes to one page module.
All data access lives in ``data_access``; all styling in ``theme``; all
shared widgets in ``components``. Each page exposes a single ``render``.
"""

from __future__ import annotations

import streamlit as st

from wildlife_monitor.dashboard import data_access as da
from wildlife_monitor.dashboard.theme import inject_css, BLACK, GREY, CARD_BORDER
from wildlife_monitor.dashboard.views import (
    overview, upload, detection, detection_map, comparison,
    review, settings, export,
)

_PAGES = [
    "Overview", "Upload", "Species Detection", "Detection Map",
    "Pipeline Comparison", "Image Review", "Settings", "Export",
]


def main() -> None:
    st.set_page_config(page_title="Serengeti Wildlife Monitor",
                       layout="wide", initial_sidebar_state="expanded")
    inject_css()

    page, species, pipeline = _sidebar()
    _route(page, species, pipeline)


def _sidebar() -> tuple[str, str, str]:
    with st.sidebar:
        st.markdown(
            f"<div style='padding:14px 4px;border-bottom:1px solid {CARD_BORDER};"
            f"margin-bottom:14px'>"
            f"<div style='font-size:14px;font-weight:700;color:{BLACK}'>"
            f"Serengeti Monitor</div>"
            f"<div style='font-size:11px;color:{GREY};margin-top:3px'>"
            f"Pipeline 1 · Camera Trap Analysis</div></div>",
            unsafe_allow_html=True)

        page = st.radio("Navigation", _PAGES, label_visibility="collapsed")

        st.markdown(
            f"<div style='font-size:10px;font-weight:600;text-transform:uppercase;"
            f"letter-spacing:.6px;color:{GREY};margin:16px 0 6px'>Filters</div>",
            unsafe_allow_html=True)

        available = da.species_list()
        species = st.selectbox("Species", available or ["No data"],
                               format_func=da.pretty)

        pipelines = list(da.PIPELINE_DISPLAY.keys())
        pipeline = st.selectbox(
            "Pipeline", pipelines,
            format_func=lambda p: da.PIPELINE_DISPLAY[p]["label"])

        st.markdown(
            f"<div style='font-size:10px;color:{GREY};line-height:1.7;"
            f"border-top:1px solid {CARD_BORDER};padding-top:14px;"
            f"margin-top:18px'>University of Johannesburg<br>"
            f"BSc Honours · CS with AI<br>"
            f"Mintesinot Zemade Necha · 2026</div>", unsafe_allow_html=True)

    return page, species, pipeline


def _route(page: str, species: str, pipeline: str) -> None:
    if page == "Overview":
        overview.render(species)
    elif page == "Upload":
        upload.render(species)
    elif page == "Species Detection":
        detection.render(species, pipeline)
    elif page == "Detection Map":
        detection_map.render(species, pipeline)
    elif page == "Pipeline Comparison":
        comparison.render(species)
    elif page == "Image Review":
        review.render(species, pipeline)
    elif page == "Settings":
        settings.render()
    elif page == "Export":
        export.render(species)


if __name__ == "__main__":
    main()
