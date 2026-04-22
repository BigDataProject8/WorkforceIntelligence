"""Survival Curves tab.

Shows Kaplan-Meier survival curves stratified by a user-selected attribute
and annotates the plot with the log-rank test statistic so the viewer can
tell whether the gap between curves is statistically meaningful.

Alongside the plot we show the log-rank summary table produced by the
notebook — a quick reference for every stratification we studied, including
flags for protected attributes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
from lifelines.statistics import logrank_test, multivariate_logrank_test

from .ui import km_plot


# Each entry: display label → (column in the processed df, encoded→label map).
# Kept in one place so adding a new stratification is a one-line change.
_GROUP_OPTIONS: dict[str, tuple[str, dict | None]] = {
    "Overtime": ("OverTime", {0: "No Overtime", 1: "Overtime"}),
    "Satisfaction level": (
        "SatisfactionLevel",
        {0: "Satisfied (≥2)", 1: "Unsatisfied (<2)"},
    ),
    "Job level": ("JobLevelGroup", None),
    "Gender (⚠ protected)": ("GenderLabel", None),
    "Age bracket (⚠ protected)": ("AgeBracket", None),
}


def render_survival_curves(frames: dict) -> None:
    """Render the Survival Curves tab."""
    df = frames["df"]
    lr = frames["logrank"]

    st.markdown(
        "**Survival curves** show the percentage of employees still with the "
        "company as tenure grows. A curve that drops faster means people in "
        "that group leave sooner. The **log-rank test** asks whether the gap "
        "between two curves is big enough to be real rather than noise — "
        "*p < 0.05* is the common threshold for \"yes, this is a real gap.\""
    )

    choice = st.selectbox("Stratify by", list(_GROUP_OPTIONS.keys()))
    group_col, label_map = _GROUP_OPTIONS[choice]

    left, right = st.columns([2, 1])

    with left:
        _render_km_panel(df, group_col, label_map, choice)

    with right:
        _render_logrank_summary(lr)


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

def _render_km_panel(df, group_col: str, label_map: dict | None, choice: str) -> None:
    """Draw the stratified KM plot and overlay the log-rank χ² / p-value."""
    fig, ax = plt.subplots(figsize=(8, 5))
    km_plot(ax, df, group_col, label_map)

    # Compute the log-rank statistic live so the displayed number always matches
    # the plot — even when a custom stratification is added.
    groups = sorted(df[group_col].dropna().unique())
    if len(groups) == 2:
        res = logrank_test(
            df[df[group_col] == groups[0]]["SurvivalTime"],
            df[df[group_col] == groups[1]]["SurvivalTime"],
            event_observed_A=df[df[group_col] == groups[0]]["EventObserved"],
            event_observed_B=df[df[group_col] == groups[1]]["EventObserved"],
        )
        p, chi2 = res.p_value, res.test_statistic
    else:
        # Multivariate path for 3+ groups (e.g. Age brackets, Job levels).
        res = multivariate_logrank_test(
            df["SurvivalTime"].to_numpy(),
            df[group_col].to_numpy(),
            event_observed=df["EventObserved"].to_numpy(),
        )
        p, chi2 = res.p_value, res.test_statistic

    # Conventional significance stars for a quick visual read.
    sig = (
        "***" if p < 0.001
        else "**" if p < 0.01
        else "*" if p < 0.05
        else "ns"
    )
    ax.set_title(
        f"Stratified by {choice} — log-rank χ²={chi2:.1f}, p={p:.4f} {sig}"
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_logrank_summary(lr) -> None:
    """Static table of every log-rank test we ran in the notebook."""
    st.markdown("#### Log-rank summary")
    st.caption(
        "One row per stratification studied in the notebook. "
        "*Protected = True* means the split is on a legally protected attribute."
    )
    st.dataframe(
        lr.style.format({"chi2": "{:.2f}", "p": "{:.4f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "A significant result on a protected attribute isn't automatically "
        "discrimination — it's a red flag that the *data* already carries a "
        "group gap, which the model will learn from unless we intervene."
    )
