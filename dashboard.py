"""Workforce Intelligence dashboard — entry point.

Run from the repo root::

    streamlit run dashboard.py

The actual UI code lives in the ``dashboard_lib`` package so each tab stays
small and easy to navigate:

    dashboard_lib/
        config.py          — colour palette, artifact path, fairness thresholds
        artifacts.py       — cached loaders for pickled models and frames
        ui.py              — shared UI helpers and the persistent sidebar
        recommendations.py — rule-based retention action engine
        overview.py        — "Overview" tab
        risk_explorer.py   — "Risk Explorer" tab
        survival.py        — "Survival Curves" tab
        fairness.py        — "Fairness Audit" tab
        roi.py             — "ROI Calculator" tab

This entry point only wires things together — no business logic lives here.
"""
from __future__ import annotations

import streamlit as st

from dashboard_lib.artifacts import load_frames, load_models
from dashboard_lib.fairness import render_fairness_audit
from dashboard_lib.overview import render_overview
from dashboard_lib.risk_explorer import render_risk_explorer
from dashboard_lib.roi import render_roi_calculator
from dashboard_lib.survival import render_survival_curves
from dashboard_lib.ui import render_sidebar


# Streamlit page configuration must be called *before* any other Streamlit
# command, which is why it lives here at module scope.
st.set_page_config(
    page_title="Workforce Intelligence",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    """Wire the sidebar + five tabs together."""
    st.title("Workforce Intelligence")
    st.caption(
        "Predicting employee attrition with flight-risk tiers, survival "
        "curves, fairness auditing, and retention-programme ROI — IBM HR "
        "Analytics dataset."
    )

    # The sidebar is global across tabs, so it's rendered once here.
    render_sidebar()

    # Cached loaders — heavy work runs at most once per session.
    frames = load_frames()
    models = load_models()

    overview_tab, risk_tab, survival_tab, fairness_tab, roi_tab = st.tabs(
        [
            "Overview",
            "Risk Explorer",
            "Survival Curves",
            "Fairness Audit",
            "ROI Calculator",
        ]
    )
    with overview_tab:
        render_overview(frames, models)
    with risk_tab:
        render_risk_explorer(frames, models)
    with survival_tab:
        render_survival_curves(frames)
    with fairness_tab:
        render_fairness_audit(frames)
    with roi_tab:
        render_roi_calculator(frames)


if __name__ == "__main__":
    main()
