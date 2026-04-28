"""Overview tab.

Lands the user with:
  1. A *Headline findings* callout summarising the analysis in four bullets.
  2. KPI cards for the workforce.
  3. Attrition rate by department and job role (where should HR focus first?).
  4. Global feature importance — which attributes drive attrition overall.
  5. Classifier performance on the held-out test split.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .config import PALETTE
from .ui import tier_threshold_rows


def render_overview(frames: dict, models: dict) -> None:
    """Render the Overview tab into the current Streamlit container."""
    df = frames["df"]
    risk = frames["risk"]

    st.markdown(
        "This tab gives a top-down view of the workforce and the two models "
        "we run on it. Start here to understand *who* is leaving, *where* "
        "they cluster, and *how well* the LASSO classifier distinguishes "
        "leavers from stayers. The risk tiers shown throughout the dashboard "
        "are produced by a separate **fair Cox** survival model — see the "
        "tier-definition table below."
    )

    _render_headline_findings()

    st.divider()
    _render_kpis(df, risk)

    st.divider()
    _render_risk_tier_legend(risk)

    st.divider()
    _render_segment_breakdowns(df)

    st.divider()
    _render_global_feature_importance(frames, models)

    st.divider()
    _render_classifier_metrics(frames, models)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def _render_headline_findings() -> None:
    """Four-bullet summary of the analysis — the "if you only read one thing" box.

    Content distilled from the notebook's Part 5.8 Conclusion and SHAP drivers.
    Kept as a static string because these findings only change when the
    underlying modelling changes, not per-session.
    """
    st.info(
        "**Headline findings**\n\n"
        "1. **Pay and overtime are the two biggest modifiable drivers.** "
        "Under the LASSO classifier, **MonthlyIncome (log)** tops the SHAP "
        "global importance ranking — low pay carries the largest "
        "attribution toward attrition — with raw MonthlyIncome and OverTime "
        "right behind it. Overtime also shows up strongly in the Cox "
        "hazards and KM curves, so compensation review and schedule relief "
        "are the two highest-leverage retention levers.\n\n"
        "2. **The first two years are the danger zone.** Both the KM curves "
        "and at-risk tables show the steepest drop in retention in years "
        "1–2, so onboarding and early-career investment has the highest "
        "marginal value.\n\n"
        "3. **Satisfaction is a strong protective factor.** Employees with a "
        "composite satisfaction score above the midpoint show materially "
        "higher retention than those below — and *LowSatisfactionFlag* sits "
        "in the LASSO classifier's top ten drivers.\n\n"
        "4. **The fair model loses almost no accuracy.** Removing gender and "
        "age as Cox inputs barely moves the C-index — we recommend deploying "
        "the fair model as the default."
    )


def _render_kpis(df: pd.DataFrame, risk: pd.DataFrame) -> None:
    """Top-line counts: workforce size, attrition rate, median tenure, high-risk."""
    total = len(df)
    attrition_rate = df["Attrition"].mean()
    median_tenure = float(np.median(df["SurvivalTime"]))
    high_risk = int((risk["RiskTier"] == "High Risk").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Employees", f"{total:,}")
    c2.metric("Attrition rate", f"{attrition_rate * 100:.1f}%")
    c3.metric("Median tenure (yrs)", f"{median_tenure:.1f}")
    c4.metric(
        "High-risk employees",
        f"{high_risk}",
        f"{high_risk / total * 100:.1f}% of workforce",
    )


def _render_risk_tier_legend(risk: pd.DataFrame) -> None:
    """Show how risk tiers are assigned alongside the live per-tier headcount.

    Pairs the abstract probability cut-offs with concrete data — the count
    of employees in each band and the actual attrition rate observed within
    that band — so HR can sanity-check that the bands track real outcomes.
    The threshold strings come from ``tier_threshold_rows`` so this table
    stays in sync with the notebook's cut-offs.
    """
    st.subheader("How risk tiers are assigned")
    st.caption(
        "Tiers come from the **fair Cox model**'s predicted 1-year attrition "
        "probability — the survival model fitted *without* gender or age. "
        "The Cox model decides who lands in which band; the LASSO classifier "
        "supplies the SHAP-based driver attribution shown below."
    )

    # Aggregate per tier. Using a fixed reindex keeps the table in High → Low
    # order regardless of the alphabetical sort the groupby would default to.
    summary = (
        risk.groupby("RiskTier")
        .agg(
            count=("RiskTier", "size"),
            actual=("Attrition", "mean"),
            avg_pred=("AttritionProb1Yr", "mean"),
        )
        .reindex(["High Risk", "Moderate Risk", "Low Risk"])
    )

    threshold_lookup = dict(tier_threshold_rows())
    total = int(summary["count"].sum())

    table = pd.DataFrame(
        {
            "Tier": summary.index,
            "Threshold (1-yr predicted prob.)": [
                threshold_lookup[t] for t in summary.index
            ],
            "Employees": summary["count"].astype(int).map("{:,}".format),
            "Share of workforce": (summary["count"] / total * 100).map(
                "{:.1f}%".format
            ),
            "Actual attrition rate": (summary["actual"] * 100).map(
                "{:.1f}%".format
            ),
            "Avg predicted prob.": (summary["avg_pred"] * 100).map(
                "{:.1f}%".format
            ),
        }
    )
    # ``hide_index=True`` keeps the table reading as a definition lookup
    # rather than a queryable dataframe.
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(
        "A well-calibrated set of tiers should show actual attrition rising "
        "monotonically from Low → Moderate → High Risk."
    )


def _render_segment_breakdowns(df: pd.DataFrame) -> None:
    """Horizontal bar charts of attrition rate by department and job role.

    Reconstructs the categorical label from the one-hot columns (our processed
    frame drops the original string columns at encoding time). The bars are
    sorted ascending so the highest-attrition rows sit at the top, the first
    thing the eye lands on.
    """
    st.subheader("Where is attrition concentrated?")
    st.caption(
        "Attrition rate is the share of employees in each group who have left. "
        "Groups with both a high bar and a large *n* deserve the earliest attention."
    )

    seg_cols = st.columns(2)
    for ax_col, prefix, title in [
        (seg_cols[0], "Department", "Department"),
        (seg_cols[1], "JobRole", "Job role"),
    ]:
        onehot_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
        if not onehot_cols:
            continue

        # Reconstruct the categorical label from one-hot columns.
        label = df[onehot_cols].idxmax(axis=1).str.replace(f"{prefix}_", "", regex=False)
        agg = (
            pd.DataFrame({"group": label, "Attrition": df["Attrition"]})
            .groupby("group")["Attrition"]
            .agg(["mean", "size"])
            .sort_values("mean", ascending=True)
        )

        fig, ax = plt.subplots(figsize=(6, max(3, 0.35 * len(agg))))
        ax.barh(agg.index, agg["mean"] * 100, color=PALETTE[0], alpha=0.85)
        for i, (rate, n) in enumerate(zip(agg["mean"], agg["size"])):
            ax.text(rate * 100 + 0.3, i, f"n={n}", va="center", fontsize=8, color="#555")
        ax.set(xlabel="Attrition rate (%)", title=f"Attrition by {title}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        ax_col.pyplot(fig)
        plt.close(fig)


def _render_global_feature_importance(frames: dict, models: dict) -> None:
    """Global feature importance as mean-|SHAP| across the test set.

    The per-employee Risk Explorer answers "why this person?"; this section
    answers "what drives attrition *overall*?" by averaging absolute SHAP
    values across everyone in the held-out test split. We pre-compute the
    SHAP values at export time, so this panel is just a sort + plot.
    """
    st.subheader("What drives attrition overall?")
    st.caption(
        "Average of the absolute SHAP contribution per feature across the "
        "test split. A feature at the top influenced the classifier the most, "
        "regardless of which direction it pushed for each individual."
    )

    shap_test = models["shap_test"]
    feature_names = frames["X_test"].columns.tolist()

    # Mean absolute SHAP per feature — standard global-importance summary.
    mean_abs = np.abs(shap_test).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=True)
        .tail(15)
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance["feature"], importance["importance"], color=PALETTE[4], alpha=0.85)
    ax.set(
        xlabel="Mean |SHAP value| (log-odds impact on attrition)",
        title="Top 15 attrition drivers across the workforce",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_classifier_metrics(frames: dict, models: dict) -> None:
    """Headline metrics for the class-weighted LASSO classifier."""
    st.subheader("Classifier performance — LASSO Logistic Regression (class-weighted)")
    st.caption(
        "These numbers describe how well the classifier separates leavers from "
        "stayers on an unseen 20% test split. **Recall** is the most operational "
        "metric — of everyone who actually leaves, what fraction did we catch?"
    )

    lasso = models["lasso"]
    scaler = models["scaler"]
    X_test, y_test = frames["X_test"], frames["y_test"]

    # LASSO was fit on standardised features, so the held-out split has to be
    # transformed with the *training-time* scaler before scoring. Using a
    # freshly-fit scaler here would re-leak test statistics into the model.
    X_test_scaled = scaler.transform(X_test)

    y_pred = lasso.predict(X_test_scaled)
    y_prob = lasso.predict_proba(X_test_scaled)[:, 1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    m4.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
    m5.metric("AUROC", f"{roc_auc_score(y_test, y_prob):.3f}")

    st.caption(
        f"Evaluated on the held-out 20% test split "
        f"(n={len(y_test)} employees, {int(y_test.sum())} positives)."
    )
