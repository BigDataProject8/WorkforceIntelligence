"""Fairness Audit tab.

Structured as a formal audit report:
  1. **Audit Summary** — traffic-light verdict on each of the four metrics
     (disparity, calibration, C-index gap) for each protected attribute.
  2. **Demographic parity** — mean predicted risk and High-Risk flagging rate
     by group, with the disparity ratio.
  3. **Calibration within risk tiers** — actual vs predicted attrition rate
     within each risk tier, per group.
  4. **Discrimination (C-index per group)** — does the model rank equally
     well for each group?

All metrics are computed against the *fair* Cox model (gender and age
excluded as inputs). We still audit because other features correlate with
protected attributes — the "proxy discrimination" problem.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from lifelines.utils import concordance_index

from .config import (
    CALIBRATION_GAP_FLAG_PP,
    C_INDEX_GAP_FLAG,
    DISPARITY_FLAG,
    PALETTE,
)


# Protected groupings we audit. Kept at module level so the summary section
# and the detail sections stay in sync without a global variable.
_GROUPINGS = [
    ("GenderLabel", ["Female", "Male"], "Gender"),
    ("AgeBracket", ["Young (<35)", "Mid (35-49)", "Senior (50+)"], "Age bracket"),
]

_TIER_ORDER = ["High Risk", "Moderate Risk", "Low Risk"]


def render_fairness_audit(frames: dict) -> None:
    """Render the Fairness Audit tab."""
    risk = frames["risk"]

    st.markdown(
        "This tab audits the **fairness-constrained Cox model** (gender and "
        "age excluded as inputs) against four fairness metrics. We keep the "
        "audit because other features — income, job level, tenure — correlate "
        "with protected attributes and can reconstruct demographic patterns "
        "even when the attributes themselves aren't in the model."
    )

    _render_audit_summary(risk)

    st.divider()
    _render_parity_section(risk)

    st.divider()
    _render_calibration_section(risk)

    st.divider()
    _render_c_index_section(risk)


# ---------------------------------------------------------------------------
# Audit Summary (the "headline report" view)
# ---------------------------------------------------------------------------

def _render_audit_summary(risk: pd.DataFrame) -> None:
    """Top-of-tab scorecard: one row per protected attribute × three checks.

    Gives a decision-maker a single-screen verdict before they dive into the
    detail sections below.
    """
    st.subheader("Audit summary")
    st.caption(
        "Each row is one check against one protected attribute. **PASS** means "
        "the metric is within its common fairness threshold; **FLAG** means it "
        "warrants review before the model is used for individual decisions."
    )

    rows = []
    for group_col, values, label in _GROUPINGS:
        # --- Disparity ratio (mean predicted risk max/min across groups) ---
        means = [
            risk.loc[risk[group_col] == v, "AttritionProb1Yr"].mean()
            for v in values
            if (risk[group_col] == v).sum() > 0
        ]
        disparity = (max(means) / min(means)) if means else float("nan")
        disparity_status = (
            "FLAG" if disparity > DISPARITY_FLAG else "PASS"
        )

        # --- Calibration gap (largest within-tier miscalibration in pp) ---
        max_gap = 0.0
        for v in values:
            sub = risk[risk[group_col] == v]
            for tier in _TIER_ORDER:
                t = sub[sub["RiskTier"] == tier]
                if t.empty:
                    continue
                max_gap = max(
                    max_gap,
                    abs(t["Attrition"].mean() - t["AttritionProb1Yr"].mean()) * 100,
                )
        calibration_status = (
            "FLAG" if max_gap > CALIBRATION_GAP_FLAG_PP else "PASS"
        )

        # --- C-index gap between subgroups ---
        c_values = []
        for v in values:
            sub = risk[risk[group_col] == v]
            if len(sub) < 20:
                continue
            c_values.append(
                concordance_index(
                    sub["SurvivalTime"].to_numpy(),
                    -sub["CoxHazardScore"].to_numpy(),
                    sub["EventObserved"].to_numpy(),
                )
            )
        c_gap = (max(c_values) - min(c_values)) if len(c_values) >= 2 else 0.0
        c_status = "FLAG" if c_gap > C_INDEX_GAP_FLAG else "PASS"

        rows.append(
            {
                "Protected attribute": label,
                "Disparity ratio": f"{disparity:.2f}",
                "Disparity": disparity_status,
                "Calibration max gap": f"{max_gap:.1f}pp",
                "Calibration": calibration_status,
                "C-index gap": f"{c_gap:.3f}",
                "Discrimination": c_status,
            }
        )

    summary_df = pd.DataFrame(rows)

    # Colour the verdict cells so PASS/FLAG jump out.
    def _colour(val: str) -> str:
        if val == "FLAG":
            return "background-color: #FFEBEE; color: #B71C1C; font-weight: 600"
        if val == "PASS":
            return "background-color: #E8F5E9; color: #1B5E20; font-weight: 600"
        return ""

    styled = summary_df.style.map(
        _colour, subset=["Disparity", "Calibration", "Discrimination"]
    )
    st.dataframe(styled, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------

def _render_parity_section(risk: pd.DataFrame) -> None:
    """Per-group mean predicted risk, actual attrition rate, and High-Risk share."""
    st.subheader("Demographic parity")
    st.caption(
        "The model passes **demographic parity** when predicted risk is "
        "similar across groups. The **disparity ratio** (max / min mean "
        "predicted probability) is flagged above 1.25."
    )

    for group_col, _values, label in _GROUPINGS:
        agg = (
            risk.groupby(group_col)
            .agg(
                n=("Attrition", "size"),
                ActualAttrRate=("Attrition", "mean"),
                MeanPredProb=("AttritionProb1Yr", "mean"),
                HighRiskRate=("RiskTier", lambda s: (s == "High Risk").mean()),
            )
            .reset_index()
        )
        disparity = agg["MeanPredProb"].max() / agg["MeanPredProb"].min()
        verdict = (
            "🚩 above 1.25 threshold"
            if disparity > DISPARITY_FLAG
            else "✅ within threshold"
        )

        st.markdown(
            f"**{label}** — disparity ratio (max/min mean pred prob): "
            f"`{disparity:.3f}` {verdict}"
        )

        # Format as strings so we can round to the precision the reader cares
        # about without losing sortability inside a single session.
        display = agg.copy()
        display["ActualAttrRate"] = (
            (display["ActualAttrRate"] * 100).round(1).astype(str) + "%"
        )
        display["MeanPredProb"] = display["MeanPredProb"].round(3)
        display["HighRiskRate"] = (
            (display["HighRiskRate"] * 100).round(1).astype(str) + "%"
        )
        st.dataframe(display, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _render_calibration_section(risk: pd.DataFrame) -> None:
    """Actual vs predicted attrition rate within each risk tier, per group."""
    st.subheader("Calibration within risk tiers")
    st.caption(
        "Within each risk tier (High / Moderate / Low), does the model's "
        "predicted probability match the actual attrition rate? A large gap "
        "for a specific subgroup means the model is unreliable for that group "
        "— dangerous because it still looks accurate overall."
    )

    for group_col, values, label in _GROUPINGS:
        fig, axes = plt.subplots(
            1, len(values), figsize=(4.5 * len(values), 4), sharey=True
        )
        if len(values) == 1:
            axes = [axes]

        for ax, gval in zip(axes, values):
            sub = risk[risk[group_col] == gval]
            actual, predicted = [], []
            # Iterate tiers in a fixed order so the three bar triplets line up
            # left→right consistently across all subplots.
            for tier in _TIER_ORDER:
                t = sub[sub["RiskTier"] == tier]
                if t.empty:
                    actual.append(0.0)
                    predicted.append(0.0)
                else:
                    actual.append(t["Attrition"].mean())
                    predicted.append(t["AttritionProb1Yr"].mean())

            x = np.arange(3)
            ax.bar(
                x - 0.18,
                [a * 100 for a in actual],
                0.35,
                label="Actual",
                color=PALETTE[0],
                alpha=0.85,
            )
            ax.bar(
                x + 0.18,
                [p * 100 for p in predicted],
                0.35,
                label="Predicted",
                color="#607D8B",
                alpha=0.75,
                hatch="//",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(["High", "Mod.", "Low"], fontsize=9)
            ax.set_title(f"{gval} (n={len(sub)})")
            ax.set_ylim(0, 80)
            ax.legend(fontsize=8)

            # Annotate the largest actual-vs-predicted gap so the reader can
            # spot miscalibration without eyeballing the bars.
            max_gap = max(abs(a - p) * 100 for a, p in zip(actual, predicted))
            tag_color = "darkred" if max_gap > CALIBRATION_GAP_FLAG_PP else "darkgreen"
            ax.text(
                0.5,
                0.95,
                f"Max gap: {max_gap:.1f}pp",
                transform=ax.transAxes,
                ha="center",
                va="top",
                color=tag_color,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85),
            )

        axes[0].set_ylabel("Attrition rate (%)")
        fig.suptitle(f"Calibration by {label}", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ---------------------------------------------------------------------------
# C-index per group
# ---------------------------------------------------------------------------

def _render_c_index_section(risk: pd.DataFrame) -> None:
    """Does the model rank employees equally well for every demographic group?"""
    st.subheader("Discrimination — C-index per group")
    st.caption(
        "The **C-index** measures how well the model *ranks* employees by "
        "risk. A gap greater than 0.05 between groups means the model is "
        "more accurate for one group than another — HR actions based on its "
        "top-N list would miss-target the lower-performing group."
    )

    # Overall C-index as a reference point for the per-group values.
    c_all = concordance_index(
        risk["SurvivalTime"].to_numpy(),
        -risk["CoxHazardScore"].to_numpy(),
        risk["EventObserved"].to_numpy(),
    )

    for group_col, values, label in _GROUPINGS:
        rows = []
        for gval in values:
            sub = risk[risk[group_col] == gval]
            # Require a meaningful sample to report a C-index.
            if len(sub) < 20:
                rows.append(
                    {"Group": gval, "n": len(sub), "C-index": "—", "Rating": "n/a"}
                )
                continue
            c = concordance_index(
                sub["SurvivalTime"].to_numpy(),
                -sub["CoxHazardScore"].to_numpy(),
                sub["EventObserved"].to_numpy(),
            )
            rating = (
                "Excellent" if c >= 0.80
                else "Good" if c >= 0.70
                else "Moderate" if c >= 0.60
                else "Poor"
            )
            rows.append(
                {"Group": gval, "n": len(sub), "C-index": f"{c:.3f}", "Rating": rating}
            )

        numeric = [float(r["C-index"]) for r in rows if r["C-index"] != "—"]
        gap = (max(numeric) - min(numeric)) if len(numeric) >= 2 else 0.0
        verdict = (
            "🚩 gap > 0.05" if gap > C_INDEX_GAP_FLAG else "✅ gap ≤ 0.05"
        )

        st.markdown(
            f"**{label}** — overall C-index: `{c_all:.3f}` • "
            f"subgroup gap: `{gap:.4f}` {verdict}"
        )
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
