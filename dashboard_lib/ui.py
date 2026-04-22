"""Shared UI primitives: coloured risk badges, one-hot decoding, the Kaplan-
Meier plotting helper, and the persistent sidebar with a plain-English glossary.

The sidebar is rendered once from the entry point so it stays visible across
tab switches (Streamlit's sidebar is global — rendering it from inside a tab
would make it disappear when the user navigates away).
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from lifelines import KaplanMeierFitter

from .config import PALETTE, TIER_COLORS


def tier_badge(tier: str) -> str:
    """Return an inline HTML pill coloured by risk tier.

    We use inline HTML (and ``unsafe_allow_html=True`` at the render site)
    because Streamlit's built-in widgets don't provide coloured badges.
    """
    color = TIER_COLORS.get(tier, "#607D8B")
    return (
        f"<span style='background:{color};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.9rem'>{tier}</span>"
    )


def decode_onehot(row: pd.Series, prefix: str) -> str:
    """Reconstruct a categorical value from the one-hot columns for a single row.

    Example: given ``prefix='Department'`` and a row where
    ``Department_Sales == 1``, returns ``'Sales'``. Returns ``'—'`` if nothing
    is set (shouldn't happen for a properly one-hot-encoded frame, but keeps
    the UI robust).
    """
    cols = [c for c in row.index if c.startswith(f"{prefix}_")]
    hit = [c[len(prefix) + 1:] for c in cols if row[c] == 1]
    return hit[0] if hit else "—"


def km_plot(ax, df: pd.DataFrame, group_col: str, label_map: dict | None = None) -> None:
    """Draw Kaplan-Meier survival curves for each level of ``group_col``.

    ``label_map`` supplies human-readable labels for encoded values
    (e.g. ``{0: 'No Overtime', 1: 'Overtime'}``). The 50% reference line marks
    where half the cohort has left, giving a visual anchor for median tenure.
    """
    label_map = label_map or {}
    groups = sorted(df[group_col].dropna().unique())
    for i, g in enumerate(groups):
        sub = df[df[group_col] == g]
        label = f"{label_map.get(g, g)} (n={len(sub)})"
        KaplanMeierFitter(label=label).fit(
            sub["SurvivalTime"].to_numpy(),
            event_observed=sub["EventObserved"].to_numpy(),
        ).plot_survival_function(
            ax=ax, color=PALETTE[i % len(PALETTE)], ci_show=True, ci_alpha=0.10
        )
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set(xlabel="Years at Company", ylabel="S(t)", ylim=(0, 1.05))
    ax.legend(fontsize=8)


# Glossary content kept as a module-level constant so the sidebar function
# stays compact. Terms are phrased for an HR/business reader, not a statistician.
_GLOSSARY = """
**Attrition** — an employee leaves the company.

**Flight risk / risk tier** — this model's estimate of how likely someone is
to leave in the next year. *High Risk* means >35% predicted probability,
*Moderate Risk* 15–35%, *Low Risk* <15%.

**Cox Proportional Hazards** — a survival model that estimates *when* each
employee is likely to leave, not just whether.

**Kaplan-Meier curve** — the percentage of employees still employed over time.
Steeper = faster attrition.

**Log-rank test** — a hypothesis test: "are two survival curves really
different, or could the gap be noise?" *p < 0.05* means the gap is unlikely
to be coincidence.

**SHAP** — per-employee feature attributions. A positive SHAP value for
*OverTime* means overtime is pushing the model toward predicting this
employee will leave.

**C-index (concordance)** — how well the model ranks employees by risk.
0.5 = random, 1.0 = perfect. Above 0.70 is useful.

**Calibration** — when the model says "30% chance," do 30% of those employees
actually leave? Good calibration = yes.

**Disparity ratio** — highest group-average predicted risk divided by the
lowest. Above 1.25 is the common fairness red flag.

**Fair model** — the Cox model fit *without* gender or age as inputs. We
still audit it for bias because other features (income, job level) correlate
with protected attributes.
"""


def render_sidebar() -> None:
    """Render the persistent sidebar: About + glossary + footer caption.

    Kept deliberately short — the glossary lives inside an expander so the
    default view is not overwhelming.
    """
    with st.sidebar:
        st.markdown("### About this dashboard")
        st.markdown(
            "A workforce-intelligence tool that predicts which employees are at "
            "risk of leaving, explains *why*, suggests retention actions, audits "
            "the model for demographic fairness, and estimates the ROI of "
            "targeted interventions."
        )
        st.markdown(
            "**Data:** 1,470 employees from the public IBM HR Analytics "
            "Employee Attrition dataset. No real individuals are identified."
        )
        with st.expander("Glossary — plain English"):
            st.markdown(_GLOSSARY)
        st.caption(
            "Built with Streamlit · lifelines · XGBoost · SHAP. Re-run the "
            "notebook's export cell and refresh to update the artifacts."
        )
