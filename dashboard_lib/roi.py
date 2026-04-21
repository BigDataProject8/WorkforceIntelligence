"""ROI Calculator tab.

Estimates the net financial impact of a targeted retention programme.

Model
-----
  Let p_i    = predicted 1-year attrition probability for employee i.
      s_i    = annual salary for employee i (MonthlyIncome × 12).
      r      = replacement-cost multiplier (typically 0.5–2.0× annual salary).
      e      = intervention effectiveness (fractional reduction in p_i).
      c      = cost per intervention per employee.

  baseline_cost     = Σ p_i · s_i · r                        # cost if we do nothing
  post_cost         = Σ p_i · (1 − e) · s_i · r              # cost after intervention
  intervention_cost = n_cohort · c                           # programme budget
  gross_savings     = baseline_cost − post_cost              # turnover averted
  net_savings       = gross_savings − intervention_cost      # after spend
  roi_pct           = net_savings / intervention_cost × 100  # return per dollar

Defaults
--------
  Effectiveness 30% — conservative based on published retention-programme
  meta-analyses (actual numbers vary widely by intervention type).
  Replacement cost 1.0× annual salary — middle of the 50–200% range commonly
  cited in HR research.

The numbers are sensitive to the assumption sliders: the purpose of this tab
is *scenario planning*, not a predictive forecast.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


def render_roi_calculator(frames: dict) -> None:
    """Render the ROI Calculator tab."""
    risk = frames["risk"]
    df = frames["df"]

    st.markdown(
        "Estimate the financial return on a targeted retention programme. "
        "Pick which risk tiers to intervene on, set your assumptions about "
        "programme effectiveness and cost, and the calculator projects the "
        "cost of doing nothing vs. the cost of intervening. All figures are "
        "in annualised USD."
    )

    # ---- Inputs ---------------------------------------------------------
    cohort, effectiveness, cost_per_emp, replacement_mult = _render_inputs(risk)

    if cohort.empty:
        st.warning(
            "No employees match the selected cohort. Widen the filters above."
        )
        return

    # ---- Compute financial impact --------------------------------------
    # Join cohort (which carries probabilities + tiers) to df (which carries
    # MonthlyIncome) on the shared row index so we can compute salary-based
    # costs per employee.
    cohort_full = cohort.join(
        df[["MonthlyIncome"]], how="left"
    )
    cohort_full["AnnualSalary"] = cohort_full["MonthlyIncome"] * 12

    baseline = (
        cohort_full["AttritionProb1Yr"]
        * cohort_full["AnnualSalary"]
        * replacement_mult
    )
    post = (
        cohort_full["AttritionProb1Yr"]
        * (1 - effectiveness)
        * cohort_full["AnnualSalary"]
        * replacement_mult
    )

    baseline_cost = float(baseline.sum())
    post_cost = float(post.sum())
    intervention_cost = float(len(cohort_full) * cost_per_emp)
    gross_savings = baseline_cost - post_cost
    net_savings = gross_savings - intervention_cost
    # Guard against divide-by-zero when the user zeros out the budget.
    roi_pct = (net_savings / intervention_cost * 100) if intervention_cost > 0 else 0.0

    expected_leavers_base = float(cohort_full["AttritionProb1Yr"].sum())
    expected_leavers_post = float(
        (cohort_full["AttritionProb1Yr"] * (1 - effectiveness)).sum()
    )

    # ---- KPI row --------------------------------------------------------
    _render_kpi_row(
        cohort_size=len(cohort_full),
        expected_leavers_base=expected_leavers_base,
        expected_leavers_post=expected_leavers_post,
        net_savings=net_savings,
        roi_pct=roi_pct,
    )

    st.divider()

    # ---- Cost comparison chart ------------------------------------------
    _render_cost_chart(
        baseline_cost=baseline_cost,
        post_cost=post_cost,
        intervention_cost=intervention_cost,
        net_savings=net_savings,
    )

    st.divider()

    # ---- Sensitivity ---------------------------------------------------
    _render_sensitivity(cohort_full, cost_per_emp, replacement_mult)

    # ---- Assumptions footer --------------------------------------------
    st.caption(
        f"**Assumptions in this scenario:** "
        f"cohort of {len(cohort_full)} employees · "
        f"intervention effectiveness {effectiveness * 100:.0f}% · "
        f"cost per employee ${cost_per_emp:,.0f} · "
        f"replacement cost {replacement_mult:.1f}× annual salary. "
        "Published HR research places replacement cost between 0.5× and 2.0× "
        "annual salary, varying by role seniority. Treat outputs as scenario "
        "planning, not forecast."
    )


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def _render_inputs(
    risk: pd.DataFrame,
) -> tuple[pd.DataFrame, float, float, float]:
    """Render the four input controls and return the selected cohort + assumptions.

    Returned tuple: (cohort dataframe, effectiveness, cost_per_employee,
    replacement_multiplier).
    """
    st.subheader("Scenario inputs")

    c1, c2 = st.columns(2)

    with c1:
        tiers = st.multiselect(
            "Target risk tiers",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"],
            help="Which employees receive the intervention. High Risk only is "
            "the most focused option; adding Moderate Risk expands coverage.",
        )
        effectiveness = st.slider(
            "Intervention effectiveness",
            min_value=0.0,
            max_value=0.8,
            value=0.30,
            step=0.05,
            format="%.0f%%",
            help="How much the intervention reduces each employee's attrition "
            "probability. 30% is a conservative default — highly-targeted "
            "programmes can reach 50%, broad ones often under 20%.",
        )

    with c2:
        cost_per_emp = st.number_input(
            "Cost per employee ($)",
            min_value=0,
            max_value=25_000,
            value=2_000,
            step=250,
            help="Budgeted cost per employee in the cohort. Includes any "
            "combination of 1:1 coaching, training, equity, or compensation "
            "adjustments.",
        )
        replacement_mult = st.slider(
            "Replacement cost (× annual salary)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Industry research estimates the cost of replacing an "
            "employee at 50–200% of their annual salary, driven by "
            "recruitment, onboarding, and productivity ramp-up.",
        )

    cohort = risk.loc[risk["RiskTier"].isin(tiers)].copy()
    return cohort, float(effectiveness), float(cost_per_emp), float(replacement_mult)


# ---------------------------------------------------------------------------
# Output sections
# ---------------------------------------------------------------------------

def _render_kpi_row(
    *,
    cohort_size: int,
    expected_leavers_base: float,
    expected_leavers_post: float,
    net_savings: float,
    roi_pct: float,
) -> None:
    """The main outputs: cohort size, avoided departures, net savings, ROI %."""
    st.subheader("Projected impact")

    avoided = expected_leavers_base - expected_leavers_post

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cohort size", f"{cohort_size:,}")
    c2.metric(
        "Expected leavers — baseline",
        f"{expected_leavers_base:.1f}",
        help="Sum of predicted 1-year attrition probabilities across the cohort.",
    )
    c3.metric(
        "Expected leavers — with intervention",
        f"{expected_leavers_post:.1f}",
        f"−{avoided:.1f} avoided",
    )
    c4.metric(
        "Net savings",
        f"${net_savings:,.0f}",
        help="Gross turnover averted minus programme cost.",
    )
    c5.metric(
        "ROI",
        f"{roi_pct:+.0f}%",
        help="Net savings divided by intervention spend.",
    )


def _render_cost_chart(
    *,
    baseline_cost: float,
    post_cost: float,
    intervention_cost: float,
    net_savings: float,
) -> None:
    """Side-by-side: cost of doing nothing vs cost after the programme."""
    st.subheader("Where the money goes")

    # Two stacked bars — "Do nothing" is pure turnover cost;
    # "Intervene" splits into remaining turnover cost + intervention spend,
    # with a green arrow showing the net savings.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Bar 1: status-quo cost (everything is turnover cost).
    ax.bar(
        "Do nothing",
        baseline_cost,
        color="#F44336",
        alpha=0.85,
        label="Turnover cost",
        edgecolor="white",
    )

    # Bar 2: post-intervention stack. Residual turnover at the bottom,
    # intervention spend on top — matching how the money is actually spent.
    ax.bar(
        "Intervene",
        post_cost,
        color="#F44336",
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        "Intervene",
        intervention_cost,
        bottom=post_cost,
        color="#2196F3",
        alpha=0.85,
        label="Intervention cost",
        edgecolor="white",
    )

    # Value labels on top of each bar so the viewer doesn't have to read the axis.
    total_intervene = post_cost + intervention_cost
    ax.text(0, baseline_cost, f" ${baseline_cost:,.0f}", va="bottom", ha="center", fontsize=10)
    ax.text(1, total_intervene, f" ${total_intervene:,.0f}", va="bottom", ha="center", fontsize=10)

    ax.set_ylabel("Annual cost (USD)")
    ax.set_title(
        f"Net savings: ${net_savings:,.0f}"
        + ("  (positive ROI)" if net_savings > 0 else "  (negative ROI)")
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _render_sensitivity(
    cohort_full: pd.DataFrame,
    cost_per_emp: float,
    replacement_mult: float,
) -> None:
    """How does ROI move if the effectiveness assumption is wrong?"""
    st.subheader("Sensitivity to effectiveness")
    st.caption(
        "Effectiveness is the biggest judgement call in the scenario. This "
        "chart sweeps it from 0% to 80% while holding cost and replacement "
        "multiplier fixed, so you can see where the programme breaks even."
    )

    # Sweep effectiveness in 5% steps and compute the resulting ROI %.
    levels = [i / 100 for i in range(0, 81, 5)]
    rois = []
    base_turnover = (
        cohort_full["AttritionProb1Yr"]
        * cohort_full["MonthlyIncome"]
        * 12
        * replacement_mult
    ).sum()
    spend = len(cohort_full) * cost_per_emp

    for e in levels:
        saved = base_turnover * e
        net = saved - spend
        rois.append((net / spend * 100) if spend > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot([l * 100 for l in levels], rois, marker="o", color="#2196F3", lw=2)
    ax.axhline(0, color="gray", ls="--", lw=1, label="Break-even")
    ax.set_xlabel("Intervention effectiveness (%)")
    ax.set_ylabel("ROI (%)")
    ax.set_title("Break-even effectiveness at this spend")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Find and show the break-even effectiveness — the point where ROI crosses 0.
    break_even = next(
        (l * 100 for l, r in zip(levels, rois) if r >= 0),
        None,
    )
    if break_even is not None:
        st.markdown(
            f"**Break-even effectiveness:** approximately "
            f"**{break_even:.0f}%** at this cohort size and budget."
        )
    else:
        st.markdown(
            "**Break-even effectiveness:** not reached within 0–80% at this "
            "spend. Consider narrowing the cohort or reducing per-employee "
            "cost."
        )
