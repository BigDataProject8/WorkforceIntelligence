"""Risk Explorer tab.

Lets a user drill into an individual employee:
  - Select from the filtered cohort (filters live in the tab's own sidebar block).
  - See predicted risk tier, one-year probability, outcome, and key attributes.
  - Inspect the Cox model's individual survival curve.
  - Inspect the LASSO classifier's per-employee SHAP decomposition.
  - Read rule-based retention recommendations, ranked by SHAP impact.

All four visuals share a single selected employee, so managers get a consistent
picture of *who this person is*, *what the models see*, and *what to do*.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .artifacts import build_shap_explainer
from .config import PALETTE
from .recommendations import build_recommendations
from .ui import decode_onehot, tier_badge, tier_thresholds_markdown


def render_risk_explorer(frames: dict, models: dict) -> None:
    """Render the Risk Explorer tab."""
    df = frames["df"]
    risk = frames["risk"]
    X_full = frames["X_full"]
    cox_fair = frames["cox_fair"]
    cph_fair = models["cph_fair"]

    st.markdown(
        "Pick any employee to see their predicted attrition risk, the "
        "features driving that risk, and suggested retention actions. The "
        "**risk tier** and **1-year attrition probability** come from the "
        "fair Cox survival model; the **SHAP driver chart** is generated "
        "from the LASSO classifier. Filters in the sidebar narrow the "
        "selector."
    )

    # Tier thresholds are repeated here (and not just on the Overview tab)
    # because managers tend to land directly on the Risk Explorer when they
    # have a specific employee in mind, and need to know what "High Risk"
    # actually means without flipping tabs.
    with st.expander("How risk tiers are assigned"):
        st.markdown(tier_thresholds_markdown())

    # Sidebar filters scoped to this tab. Streamlit keeps them visible whenever
    # this tab is active and hides them when the user switches away.
    filters = _render_tab_filters(df)

    # Build the selectable cohort given the current filters.
    candidates = _filter_candidates(risk, df, filters)
    if candidates.empty:
        st.warning("No employees match the selected filters.")
        return

    emp_id = _employee_selector(candidates)
    emp_row = df.iloc[emp_id]
    emp_risk = risk.loc[emp_id]

    _render_employee_header(emp_row, emp_risk)

    st.divider()

    # SHAP values are needed twice: to draw the feature-importance plot *and*
    # to rank the retention recommendations. Compute once, share across both.
    shap_series = _compute_shap(models, X_full, emp_id)

    left, right = st.columns(2)
    with left:
        _render_survival_curve(cph_fair, cox_fair, emp_id, emp_row)
    with right:
        _render_shap_panel(shap_series)

    st.divider()
    _render_recommendations(emp_row, emp_risk, shap_series, df)


# ---------------------------------------------------------------------------
# Sub-sections
# ---------------------------------------------------------------------------

def _render_tab_filters(df: pd.DataFrame) -> dict:
    """Sidebar filters for the Risk Explorer.

    Returns a dict of every filter's current value. Kept in the sidebar (not
    the main area) to leave the centre column for the employee detail view.

    Three additional filters beyond the basic tier selector keep the 1,470-row
    employee dropdown navigable:
      - Department (derived from one-hot columns)
      - Overtime  (All / Yes / No)
      - Tenure range (years at company)
    """
    # Reconstruct department labels from the one-hot columns so the user can
    # pick by display name rather than having to think in column prefixes.
    dept_cols = [c for c in df.columns if c.startswith("Department_")]
    dept_options = sorted(c.replace("Department_", "") for c in dept_cols)

    tenure_min = int(df["YearsAtCompany"].min())
    tenure_max = int(df["YearsAtCompany"].max())

    with st.sidebar:
        st.markdown("### Risk Explorer filters")
        tier_filter = st.multiselect(
            "Risk tier",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk", "Low Risk"],
        )
        dept_filter = st.multiselect(
            "Department",
            options=dept_options,
            default=dept_options,
        )
        overtime_filter = st.radio(
            "Overtime",
            options=["All", "Overtime only", "No overtime only"],
            index=0,
            horizontal=True,
        )
        tenure_range = st.slider(
            "Years at company",
            min_value=tenure_min,
            max_value=tenure_max,
            value=(tenure_min, tenure_max),
        )
        sort_mode = st.selectbox(
            "Sort employees by",
            ["Predicted risk (desc)", "Predicted risk (asc)", "Row index"],
        )
    return {
        "tiers": tier_filter,
        "departments": dept_filter,
        "overtime": overtime_filter,
        "tenure_range": tenure_range,
        "sort_mode": sort_mode,
    }


def _filter_candidates(
    risk: pd.DataFrame, df: pd.DataFrame, filters: dict
) -> pd.DataFrame:
    """Apply tier + department + overtime + tenure filters, then sort."""
    # Start with the tier filter on the risk frame.
    mask = risk["RiskTier"].isin(filters["tiers"])

    # Department is encoded one-hot in df; accept rows where any selected
    # department column is 1.
    if filters["departments"]:
        dept_cols = [f"Department_{d}" for d in filters["departments"]]
        dept_cols = [c for c in dept_cols if c in df.columns]
        if dept_cols:
            dept_mask = (df[dept_cols].sum(axis=1) > 0).reindex(risk.index, fill_value=False)
            mask &= dept_mask
    else:
        # Empty dept list = "no departments selected" = show nothing.
        mask &= pd.Series(False, index=risk.index)

    # Overtime filter — radio so it's always one of three states.
    if filters["overtime"] == "Overtime only":
        mask &= df["OverTime"].reindex(risk.index) == 1
    elif filters["overtime"] == "No overtime only":
        mask &= df["OverTime"].reindex(risk.index) == 0

    # Tenure range filter (inclusive on both ends).
    lo, hi = filters["tenure_range"]
    tenure = df["YearsAtCompany"].reindex(risk.index)
    mask &= (tenure >= lo) & (tenure <= hi)

    candidates = risk.loc[mask].copy()
    # Preserve the original row index as an explicit column so we can read it
    # back out of the selectbox label later.
    candidates["row"] = candidates.index

    sort_mode = filters["sort_mode"]
    if sort_mode == "Predicted risk (desc)":
        return candidates.sort_values("AttritionProb1Yr", ascending=False)
    if sort_mode == "Predicted risk (asc)":
        return candidates.sort_values("AttritionProb1Yr", ascending=True)
    return candidates.sort_values("row")


def _employee_selector(candidates: pd.DataFrame) -> int:
    """Render the employee dropdown and return the selected row index."""
    options = [
        f"#{int(r.row):04d} — {r.RiskTier} ({r.AttritionProb1Yr * 100:.1f}%) — "
        f"{'Left' if r.Attrition else 'Still employed'}"
        for r in candidates.itertuples()
    ]
    choice = st.selectbox("Select an employee", options, index=0)
    # The leading "#NNNN" segment carries the row index — parse it back out.
    return int(choice.split("—")[0].strip().lstrip("#"))


def _render_employee_header(emp_row: pd.Series, emp_risk: pd.Series) -> None:
    """Four-column KPI header + a single-line attribute summary."""
    header = st.columns([1, 1, 1, 1])
    header[0].markdown(
        f"**Risk tier**<br>{tier_badge(emp_risk['RiskTier'])}",
        unsafe_allow_html=True,
    )
    header[1].metric(
        "Predicted 1-yr attrition", f"{emp_risk['AttritionProb1Yr'] * 100:.1f}%"
    )
    header[2].metric(
        "Actual outcome", "Left" if emp_risk["Attrition"] else "Still employed"
    )
    # Tenure here is ``YearsAtCompany`` (renamed ``SurvivalTime`` for the
    # survival-analysis pipeline). The dataset stores it as a whole number of
    # years, so a fresh hire reads as ``0`` — which looks like a missing-data
    # bug at first glance. Tag it with "new hire" so the reader understands
    # the model is flagging a brand-new employee, not a malformed row.
    tenure_yrs = float(emp_row["SurvivalTime"])
    if tenure_yrs < 1:
        tenure_value = "<1"
        tenure_delta = "new hire"
    else:
        tenure_value = f"{tenure_yrs:.0f}"
        tenure_delta = None

    header[3].metric(
        "Tenure (yrs)",
        tenure_value,
        delta=tenure_delta,
        delta_color="off",
        help=(
            "Years at the company. The dataset rounds to whole years, so a "
            "tenure of <1 means an employee in their first year — these "
            "tend to dominate the High Risk tier (see headline finding #2)."
        ),
    )

    st.markdown(
        f"**Department:** {decode_onehot(emp_row, 'Department')}  •  "
        f"**Role:** {decode_onehot(emp_row, 'JobRole')}  •  "
        f"**Job level:** {int(emp_row['JobLevel'])}  •  "
        f"**Overtime:** {'Yes' if emp_row['OverTime'] else 'No'}  •  "
        f"**Monthly income:** \\${emp_row['MonthlyIncome']:,.0f}"
    )


def _compute_shap(models: dict, X_full: pd.DataFrame, emp_id: int) -> pd.Series:
    """Compute SHAP values for a single employee and return as an indexed Series.

    LASSO is a linear model, so we use ``shap.LinearExplainer`` (cheap, exact
    contributions). The feature matrix has to be standardised first because
    that's the space the model was fitted in. Scaling is column-wise, so the
    feature ordering in the returned Series still matches ``X_full.columns``.
    """
    scaler = models["scaler"]
    # Standardise the full frame once and use it as the SHAP background
    # distribution. The cached explainer means this only happens once per
    # session even though we rebuild ``X_full_scaled`` per call.
    X_full_scaled = scaler.transform(X_full)
    explainer = build_shap_explainer(models["lasso"], X_full_scaled)

    # Scale just the selected row and pull its SHAP values. ``shap_values``
    # returns shape (1, n_features) for a single row; take row 0.
    row_scaled = scaler.transform(X_full.iloc[[emp_id]])
    shap_values = explainer.shap_values(row_scaled)[0]
    return pd.Series(shap_values, index=X_full.columns)


def _render_survival_curve(cph_fair, cox_fair, emp_id, emp_row) -> None:
    """Cox-predicted individual survival curve, with their observed tenure overlaid."""
    st.subheader("Predicted survival curve")
    st.caption(
        "The curve shows the model's forecast of this employee's probability "
        "of still being with the company over time. The red dotted line marks "
        "their *actual* tenure — useful as a sanity check for the model's shape."
    )

    emp_cox = cox_fair.iloc[[emp_id]].drop(columns=["SurvivalTime", "EventObserved"])
    sf = cph_fair.predict_survival_function(emp_cox)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.step(sf.index, sf.iloc[:, 0], where="post", color=PALETTE[0], lw=2)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(
        emp_row["SurvivalTime"],
        color="red",
        ls=":",
        lw=1.2,
        label=f"Observed tenure = {emp_row['SurvivalTime']:.0f}y",
    )
    ax.set(
        xlabel="Years",
        ylabel="S(t)",
        ylim=(0, 1.02),
        title="Fair Cox model — individual forecast",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_shap_panel(shap_series: pd.Series) -> None:
    """Top-12 SHAP bar chart for the selected employee."""
    st.subheader("Top drivers (SHAP, LASSO)")

    # Sort by absolute impact, keep the top 12, then re-sort by signed value so
    # the bar chart reads cleanly from negative (bottom) to positive (top).
    contrib = (
        shap_series.to_frame("shap")
        .assign(abs=lambda d: d["shap"].abs())
        .sort_values("abs", ascending=False)
        .head(12)
        .sort_values("shap")
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#F44336" if v > 0 else "#2196F3" for v in contrib["shap"]]
    ax.barh(contrib.index, contrib["shap"], color=colors)
    ax.axvline(0, color="black", lw=0.7)
    ax.set(
        xlabel="SHAP contribution (→ pushes toward attrition)",
        title="Top 12 features — this employee",
    )
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Red bars push the classifier toward *Attrition = Yes*, blue bars push "
        "toward *No*. Magnitudes are log-odds contributions."
    )


def _render_recommendations(
    emp_row: pd.Series,
    emp_risk: pd.Series,
    shap_series: pd.Series,
    df: pd.DataFrame,
) -> None:
    """Print the ranked retention actions (or an informational note)."""
    st.subheader("Recommended retention actions")

    if emp_risk["RiskTier"] == "Low Risk":
        st.success(
            "This employee is in the Low Risk tier. No targeted intervention "
            "is recommended — continue standard engagement practices."
        )
        return

    recs = build_recommendations(emp_row, shap_series, df)
    if not recs:
        st.info(
            "No rule-based triggers fired for this employee despite their "
            "elevated risk score. Review the SHAP drivers above and consider "
            "a qualitative 1:1 to surface issues not captured in the data."
        )
        return

    st.caption(
        "Actions are ranked by this employee's SHAP contribution for the "
        "underlying feature — the action at the top targets the strongest "
        "driver of their predicted flight risk."
    )
    for i, rec in enumerate(recs, start=1):
        arrow = "↑" if rec["priority"] > 0 else "↓" if rec["priority"] < 0 else "·"
        st.markdown(
            f"**{i}. {rec['headline']}**  "
            f"<span style='color:#888;font-size:0.85em'>"
            f"(driver: `{rec['feature']}` · SHAP {arrow} {rec['priority']:+.2f})</span>  \n"
            f"{rec['detail']}",
            unsafe_allow_html=True,
        )
