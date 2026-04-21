"""Workforce Intelligence dashboard.

Loads artifacts produced by the final cell of ``project-08.ipynb`` and renders:
  - Overview : headline attrition stats, department breakdowns, classifier metrics
  - Risk Explorer : per-employee Cox survival curve + SHAP explanation
  - Survival Curves : Kaplan-Meier comparisons with log-rank significance
  - Fairness Audit : parity, calibration, and C-index gaps across protected groups

Run from the repo root:  streamlit run dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
import streamlit as st
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ARTIFACTS = Path(__file__).parent / "artifacts"
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
TIER_COLORS = {"High Risk": "#F44336", "Moderate Risk": "#FF9800", "Low Risk": "#4CAF50"}

st.set_page_config(
    page_title="Workforce Intelligence",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Artifact loaders (cached so the app stays snappy across reruns)
# ---------------------------------------------------------------------------

@st.cache_data
def load_frames() -> dict[str, pd.DataFrame]:
    if not ARTIFACTS.exists():
        st.error(
            f"Missing `{ARTIFACTS.name}/` directory. Run the final export cell in "
            "`project-08.ipynb` first."
        )
        st.stop()

    return {
        "df": pl.read_parquet(ARTIFACTS / "df_processed.parquet").to_pandas(),
        "X_full": pd.read_parquet(ARTIFACTS / "X_full.parquet"),
        "y_full": pd.read_parquet(ARTIFACTS / "y_full.parquet")["Attrition"],
        "X_test": pd.read_parquet(ARTIFACTS / "X_test.parquet"),
        "y_test": pd.read_parquet(ARTIFACTS / "y_test.parquet")["Attrition"],
        "cox_fair": pd.read_parquet(ARTIFACTS / "cox_pd_fair.parquet"),
        "cox_full": pd.read_parquet(ARTIFACTS / "cox_pd_full.parquet"),
        "risk": pl.read_parquet(ARTIFACTS / "risk_scores.parquet").to_pandas(),
        "logrank": pl.read_parquet(ARTIFACTS / "logrank_results.parquet").to_pandas(),
    }


@st.cache_resource
def load_models() -> dict:
    features = json.loads((ARTIFACTS / "cox_features.json").read_text())
    return {
        "xgb": joblib.load(ARTIFACTS / "xgb_weighted.joblib"),
        "cph_fair": joblib.load(ARTIFACTS / "cph_fair.joblib"),
        "cph_full": joblib.load(ARTIFACTS / "cph_full.joblib"),
        "cox_features": features,
        "durations": np.load(ARTIFACTS / "durations.npy"),
        "events": np.load(ARTIFACTS / "events.npy"),
        "shap_test": np.load(ARTIFACTS / "shap_values_xgb_w.npy"),
    }


@st.cache_resource
def build_shap_explainer(_model) -> shap.TreeExplainer:
    return shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#607D8B")
    return (
        f"<span style='background:{color};color:white;padding:4px 12px;"
        f"border-radius:12px;font-weight:600;font-size:0.9rem'>{tier}</span>"
    )


def decode_onehot(row: pd.Series, prefix: str) -> str:
    cols = [c for c in row.index if c.startswith(f"{prefix}_")]
    hit = [c[len(prefix) + 1:] for c in cols if row[c] == 1]
    return hit[0] if hit else "—"


def km_plot(ax, df: pd.DataFrame, group_col: str, label_map: dict | None = None) -> None:
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


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

def render_overview(frames: dict, models: dict) -> None:
    df = frames["df"]
    risk = frames["risk"]

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

    st.divider()

    st.subheader("Attrition by segment")
    seg_cols = st.columns(2)

    for ax_col, prefix, title in [
        (seg_cols[0], "Department", "Department"),
        (seg_cols[1], "JobRole", "Job role"),
    ]:
        onehot_cols = [c for c in df.columns if c.startswith(f"{prefix}_")]
        if not onehot_cols:
            continue
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

    st.divider()

    st.subheader("Classifier performance — XGBoost (class-weighted)")
    xgb = models["xgb"]
    X_test, y_test = frames["X_test"], frames["y_test"]
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)[:, 1]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    m2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    m3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    m4.metric("F1", f"{f1_score(y_test, y_pred):.3f}")
    m5.metric("AUROC", f"{roc_auc_score(y_test, y_prob):.3f}")

    st.caption(
        "Metrics evaluated on the held-out 20% test split "
        f"(n={len(y_test)} employees, {int(y_test.sum())} positives)."
    )


def render_risk_explorer(frames: dict, models: dict) -> None:
    df = frames["df"]
    risk = frames["risk"]
    X_full = frames["X_full"]
    cox_fair = frames["cox_fair"]
    cph_fair = models["cph_fair"]
    fair_features = models["cox_features"]["fair"]

    with st.sidebar:
        st.markdown("### Filters")
        tier_filter = st.multiselect(
            "Risk tier",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk", "Low Risk"],
        )
        sort_mode = st.selectbox(
            "Sort employees by",
            ["Predicted risk (desc)", "Predicted risk (asc)", "Row index"],
        )

    mask = risk["RiskTier"].isin(tier_filter)
    candidates = risk.loc[mask].copy()
    candidates["row"] = candidates.index
    if sort_mode == "Predicted risk (desc)":
        candidates = candidates.sort_values("AttritionProb1Yr", ascending=False)
    elif sort_mode == "Predicted risk (asc)":
        candidates = candidates.sort_values("AttritionProb1Yr", ascending=True)
    else:
        candidates = candidates.sort_values("row")

    if candidates.empty:
        st.warning("No employees match the selected filters.")
        return

    options = [
        f"#{int(r.row):04d} — {r.RiskTier} ({r.AttritionProb1Yr * 100:.1f}%) — "
        f"{'Left' if r.Attrition else 'Still employed'}"
        for r in candidates.itertuples()
    ]
    choice = st.selectbox("Select an employee", options, index=0)
    emp_id = int(choice.split("—")[0].strip().lstrip("#"))

    emp_row = df.iloc[emp_id]
    emp_risk = risk.loc[emp_id]

    header = st.columns([1, 1, 1, 1])
    header[0].markdown(
        f"**Risk tier**<br>{tier_badge(emp_risk['RiskTier'])}",
        unsafe_allow_html=True,
    )
    header[1].metric("Predicted 1-yr attrition", f"{emp_risk['AttritionProb1Yr'] * 100:.1f}%")
    header[2].metric("Actual outcome", "Left" if emp_risk["Attrition"] else "Still employed")
    header[3].metric("Tenure (yrs)", f"{emp_row['SurvivalTime']:.0f}")

    st.markdown(
        f"**Department:** {decode_onehot(emp_row, 'Department')}  •  "
        f"**Role:** {decode_onehot(emp_row, 'JobRole')}  •  "
        f"**Job level:** {int(emp_row['JobLevel'])}  •  "
        f"**Overtime:** {'Yes' if emp_row['OverTime'] else 'No'}  •  "
        f"**Monthly income:** ${emp_row['MonthlyIncome']:,.0f}"
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Predicted survival curve")
        emp_cox = cox_fair.iloc[[emp_id]].drop(columns=["SurvivalTime", "EventObserved"])
        sf = cph_fair.predict_survival_function(emp_cox)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.step(sf.index, sf.iloc[:, 0], where="post", color=PALETTE[0], lw=2)
        ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
        ax.axvline(emp_row["SurvivalTime"], color="red", ls=":", lw=1.2,
                   label=f"Observed tenure = {emp_row['SurvivalTime']:.0f}y")
        ax.set(xlabel="Years", ylabel="S(t)", ylim=(0, 1.02),
               title="Fair Cox model — individual forecast")
        ax.legend(fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Top drivers (SHAP, XGBoost)")
        explainer = build_shap_explainer(models["xgb"])
        shap_values = explainer.shap_values(X_full.iloc[[emp_id]])[0]
        contrib = (
            pd.DataFrame({"feature": X_full.columns, "shap": shap_values})
            .assign(abs=lambda d: d["shap"].abs())
            .sort_values("abs", ascending=False)
            .head(12)
            .sort_values("shap")
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = ["#F44336" if v > 0 else "#2196F3" for v in contrib["shap"]]
        ax.barh(contrib["feature"], contrib["shap"], color=colors)
        ax.axvline(0, color="black", lw=0.7)
        ax.set(xlabel="SHAP contribution (→ pushes toward attrition)",
               title="Top 12 features — this employee")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Red bars push the classifier toward *Attrition = Yes*, "
            "blue bars push toward *No*. Magnitudes are log-odds contributions."
        )


def render_survival_curves(frames: dict) -> None:
    df = frames["df"]
    lr = frames["logrank"]

    st.subheader("Kaplan-Meier curves")

    group_options = {
        "Overtime": ("OverTime", {0: "No Overtime", 1: "Overtime"}),
        "Satisfaction level": ("SatisfactionLevel", {0: "Satisfied (≥2)", 1: "Unsatisfied (<2)"}),
        "Job level": ("JobLevelGroup", None),
        "Gender (⚠ protected)": ("GenderLabel", None),
        "Age bracket (⚠ protected)": ("AgeBracket", None),
    }
    choice = st.selectbox("Stratify by", list(group_options.keys()))
    group_col, label_map = group_options[choice]

    left, right = st.columns([2, 1])

    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        km_plot(ax, df, group_col, label_map)

        groups = sorted(df[group_col].dropna().unique())
        if len(groups) == 2:
            res = logrank_test(
                df[df[group_col] == groups[0]]["SurvivalTime"],
                df[df[group_col] == groups[1]]["SurvivalTime"],
                event_observed_A=df[df[group_col] == groups[0]]["EventObserved"],
                event_observed_B=df[df[group_col] == groups[1]]["EventObserved"],
            )
            p = res.p_value
            chi2 = res.test_statistic
        else:
            res = multivariate_logrank_test(
                df["SurvivalTime"].to_numpy(),
                df[group_col].to_numpy(),
                event_observed=df["EventObserved"].to_numpy(),
            )
            p, chi2 = res.p_value, res.test_statistic

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"Stratified by {choice} — log-rank χ²={chi2:.1f}, p={p:.4f} {sig}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("#### Log-rank summary")
        st.dataframe(
            lr.style.format({"chi2": "{:.2f}", "p": "{:.4f}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            "Rows where *Protected = True* require fairness review — raw survival "
            "gaps there will be learned by any model trained on this data."
        )


def render_fairness_audit(frames: dict, models: dict) -> None:
    risk = frames["risk"]

    st.subheader("Demographic parity")

    def summarise(group_col: str) -> pd.DataFrame:
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
        return agg

    for group_col, label in [("GenderLabel", "Gender"), ("AgeBracket", "Age bracket")]:
        agg = summarise(group_col)
        disparity = agg["MeanPredProb"].max() / agg["MeanPredProb"].min()
        flag = "🚩 above 1.25 threshold" if disparity > 1.25 else "✅ within threshold"

        st.markdown(f"**{label}**  — disparity ratio (max/min mean pred prob): `{disparity:.3f}` {flag}")

        display = agg.copy()
        display["ActualAttrRate"] = (display["ActualAttrRate"] * 100).round(1).astype(str) + "%"
        display["MeanPredProb"] = display["MeanPredProb"].round(3)
        display["HighRiskRate"] = (display["HighRiskRate"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Calibration within risk tiers")

    tier_order = ["High Risk", "Moderate Risk", "Low Risk"]

    for group_col, values, label in [
        ("GenderLabel", ["Female", "Male"], "Gender"),
        ("AgeBracket", ["Young (<35)", "Mid (35-49)", "Senior (50+)"], "Age bracket"),
    ]:
        fig, axes = plt.subplots(1, len(values), figsize=(4.5 * len(values), 4), sharey=True)
        if len(values) == 1:
            axes = [axes]
        for ax, gval in zip(axes, values):
            sub = risk[risk[group_col] == gval]
            actual, predicted = [], []
            for tier in tier_order:
                t = sub[sub["RiskTier"] == tier]
                if t.empty:
                    actual.append(0.0)
                    predicted.append(0.0)
                else:
                    actual.append(t["Attrition"].mean())
                    predicted.append(t["AttritionProb1Yr"].mean())
            x = np.arange(3)
            ax.bar(x - 0.18, [a * 100 for a in actual], 0.35, label="Actual",
                   color=PALETTE[0], alpha=0.85)
            ax.bar(x + 0.18, [p * 100 for p in predicted], 0.35, label="Predicted",
                   color="#607D8B", alpha=0.75, hatch="//")
            ax.set_xticks(x)
            ax.set_xticklabels(["High", "Mod.", "Low"], fontsize=9)
            ax.set_title(f"{gval} (n={len(sub)})")
            ax.set_ylim(0, 80)
            ax.legend(fontsize=8)
            max_gap = max(abs(a - p) * 100 for a, p in zip(actual, predicted))
            tag_color = "darkred" if max_gap > 10 else "darkgreen"
            ax.text(0.5, 0.95, f"Max gap: {max_gap:.1f}pp",
                    transform=ax.transAxes, ha="center", va="top",
                    color=tag_color, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
        axes[0].set_ylabel("Attrition rate (%)")
        fig.suptitle(f"Calibration by {label}", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.subheader("Discrimination — C-index per group")

    c_all = concordance_index(
        risk["SurvivalTime"].to_numpy(),
        -risk["CoxHazardScore"].to_numpy(),
        risk["EventObserved"].to_numpy(),
    )

    for group_col, values, label in [
        ("GenderLabel", ["Female", "Male"], "Gender"),
        ("AgeBracket", ["Young (<35)", "Mid (35-49)", "Senior (50+)"], "Age bracket"),
    ]:
        rows = []
        for gval in values:
            sub = risk[risk[group_col] == gval]
            if len(sub) < 20:
                rows.append({"Group": gval, "n": len(sub), "C-index": "—", "Rating": "n/a"})
                continue
            c = concordance_index(
                sub["SurvivalTime"].to_numpy(),
                -sub["CoxHazardScore"].to_numpy(),
                sub["EventObserved"].to_numpy(),
            )
            rating = (
                "Excellent" if c >= 0.80 else "Good" if c >= 0.70
                else "Moderate" if c >= 0.60 else "Poor"
            )
            rows.append({"Group": gval, "n": len(sub), "C-index": f"{c:.3f}", "Rating": rating})

        numeric = [float(r["C-index"]) for r in rows if r["C-index"] != "—"]
        gap = (max(numeric) - min(numeric)) if len(numeric) >= 2 else 0.0
        flag = "🚩 gap > 0.05" if gap > 0.05 else "✅ gap ≤ 0.05"

        st.markdown(f"**{label}** — overall C-index: `{c_all:.3f}` • subgroup gap: `{gap:.4f}` {flag}")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("Workforce Intelligence")
    st.caption(
        "Predicting employee attrition with flight-risk tiers, survival curves, "
        "and fairness auditing — IBM HR Analytics dataset."
    )

    frames = load_frames()
    models = load_models()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview", "Risk Explorer", "Survival Curves", "Fairness Audit"]
    )
    with tab1:
        render_overview(frames, models)
    with tab2:
        render_risk_explorer(frames, models)
    with tab3:
        render_survival_curves(frames)
    with tab4:
        render_fairness_audit(frames, models)


if __name__ == "__main__":
    main()
