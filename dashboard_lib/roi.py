"""ROI Calculator tab.

Estimates the net financial impact of a targeted retention programme.

Two cost inputs that are easy to confuse:
  • **Intervention cost per employee** ($) — what HR *spends* on each person
    in the cohort (coaching, training, comp adjustments, etc.). This is a
    flat dollar amount.
  • **Turnover cost per leaver** (× annual salary) — what the company *loses*
    when an employee walks out (recruitment, onboarding, productivity ramp).
    This is a multiplier of the leaver's annual salary, not a flat dollar.

Model
-----
  Let p_i    = predicted 1-year attrition probability for employee i.
      s_i    = annual salary for employee i (MonthlyIncome × 12).
      r      = turnover-cost multiplier (typically 0.5–2.0× annual salary).
      e      = intervention effectiveness (fractional reduction in p_i).
      c      = intervention cost per employee.

  baseline_cost     = Σ p_i · s_i · r                        # cost if we do nothing
  post_cost         = Σ p_i · (1 − e) · s_i · r              # cost after intervention
  intervention_cost = n_cohort · c                           # programme budget
  gross_savings     = baseline_cost − post_cost              # turnover averted
                    = e · baseline_cost
  net_savings       = gross_savings − intervention_cost      # after spend
  roi_pct           = net_savings / intervention_cost × 100  # return per dollar
  break_even_cpp    = gross_savings / n_cohort               # max c with ROI ≥ 0

Defaults
--------
  Effectiveness 30% — conservative based on published retention-programme
  meta-analyses (actual numbers vary widely by intervention type).
  Turnover cost 1.0× annual salary — middle of the 50–200% range commonly
  cited in HR research.

The numbers are sensitive to the assumption sliders: the purpose of this tab
is *scenario planning*, not a predictive forecast.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .config import HIGH_RISK_THRESHOLD, MODERATE_RISK_THRESHOLD


def render_roi_calculator(frames: dict) -> None:
    """Render the ROI Calculator tab."""
    risk = frames["risk"]
    df = frames["df"]

    st.markdown(
        "Estimate the financial return on a targeted retention programme. "
        "Pick which risk tiers to intervene on, set your assumptions about "
        "programme effectiveness and cost, and the calculator projects the "
        "cost of doing nothing vs. the cost of intervening. Risk tiers are "
        "the fair Cox model's 1-year attrition bands (see Overview tab for "
        "the cut-offs). All figures are in annualised USD."
    )

    # ---- Formula reference ---------------------------------------------
    # Surfaces the maths the rest of the tab is computing. Tucked in an
    # expander so it doesn't push the inputs below the fold for the common
    # case where the user just wants to play with sliders.
    _render_formula_reference()

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

    # Highest cost-per-employee that still produces non-negative ROI given
    # the current cohort, effectiveness, and turnover-cost assumptions:
    #   net_savings ≥ 0 ⇔ gross_savings ≥ n · c ⇔ c ≤ gross_savings / n
    # This is the headline number a planner needs to know "is my budget
    # realistic?" — surfaced as a KPI tile and a coloured callout below.
    break_even_cpp = (
        gross_savings / len(cohort_full) if len(cohort_full) > 0 else 0.0
    )

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
        break_even_cpp=break_even_cpp,
        cost_per_emp=cost_per_emp,
    )

    # Plain-English summary of the relationship between current spend and
    # break-even. Renders just below the KPI tiles so the callout's colour
    # (success/warning) reinforces the same signal as the ROI metric above.
    _render_break_even_callout(
        cost_per_emp=cost_per_emp,
        break_even_cpp=break_even_cpp,
        effectiveness=effectiveness,
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
        f"intervention cost \\${cost_per_emp:,.0f} per employee · "
        f"turnover cost {replacement_mult:.1f}× annual salary per leaver. "
        "Published HR research places turnover cost between 0.5× and 2.0× "
        "annual salary, varying by role seniority. Treat outputs as scenario "
        "planning, not forecast."
    )


# ---------------------------------------------------------------------------
# Formula reference
# ---------------------------------------------------------------------------

def _render_formula_reference() -> None:
    """Collapsible explainer of the ROI formula.

    Mirrors the docstring at the top of this module in user-facing form so
    the dashboard is self-documenting — a stakeholder can see exactly how
    each KPI is computed without opening the source. Closed by default to
    keep the tab compact for the common "tweak the sliders" workflow.
    """
    with st.expander("How is ROI calculated? (formula)"):
        st.markdown(
            "For each employee $i$ in the selected cohort, with predicted "
            "attrition probability $p_i$, annual salary $s_i$, "
            "**turnover-cost multiplier** $r$ (× annual salary, charged when "
            "an employee actually leaves), intervention effectiveness $e$, "
            "**intervention cost per employee** $c$ (flat dollars per "
            "person targeted), and cohort size $n$:"
        )

        # One LaTeX block per formula followed by a plain-English gloss.
        # The interleaved format keeps each definition next to its formula
        # so the reader doesn't have to map symbols to names mentally.
        st.latex(r"\text{baseline\_cost} = \sum_{i=1}^{n} p_i \cdot s_i \cdot r")
        st.markdown(
            "**Baseline cost** — expected turnover loss if we **do nothing**. "
            "Each employee's contribution is *(probability they leave) × "
            "(their salary) × (turnover-cost multiplier)*, summed across the "
            "cohort."
        )

        st.latex(r"\text{post\_cost} = \sum_{i=1}^{n} p_i \cdot (1 - e) \cdot s_i \cdot r")
        st.markdown(
            "**Post-intervention cost** — expected turnover loss **after** the "
            "programme runs. Same calculation as baseline, but every employee's "
            "attrition probability is reduced by the effectiveness factor $e$."
        )

        st.latex(r"\text{intervention\_cost} = n \cdot c")
        st.markdown(
            "**Intervention cost** — total programme spend. Flat across the "
            "cohort: number of employees targeted × cost per employee."
        )

        st.latex(
            r"\text{gross\_savings} = \text{baseline\_cost} - \text{post\_cost} "
            r"= e \cdot \sum_{i=1}^{n} p_i \cdot s_i \cdot r"
        )
        st.markdown(
            "**Gross savings** — turnover dollars **averted** by the programme, "
            "before paying for it. Equivalent to saying \"effectiveness × "
            "baseline cost\" — the share of expected losses prevented."
        )

        st.latex(r"\text{net\_savings} = \text{gross\_savings} - \text{intervention\_cost}")
        st.markdown(
            "**Net savings** — what's left after the programme pays for itself. "
            "Positive means the intervention generated more value than it cost; "
            "negative means it didn't."
        )

        st.latex(
            r"\text{ROI \%} = \frac{\text{net\_savings}}{\text{intervention\_cost}} "
            r"\times 100"
        )
        st.markdown(
            "**ROI %** — return per dollar spent, expressed as a percentage. "
            "An ROI of *+200%* means every \\$1 of programme spend returned "
            "\\$2 in averted turnover cost on top of paying itself back. *0%* "
            "is the break-even line."
        )

        st.latex(
            r"\text{break\_even\_cpp} = \frac{\text{gross\_savings}}{n} "
            r"= \frac{e \cdot \sum p_i \cdot s_i \cdot r}{n}"
        )
        st.markdown(
            "**Break-even cost per employee** — the highest value of $c$ at "
            "which net savings stay non-negative. Above it, the programme "
            "loses money no matter how it's marketed; below it, every dollar "
            "of headroom flows straight to net savings."
        )

        st.markdown(
            "Turnover cost multiplier $r$ scales each leaver's loss by an "
            "industry-typical multiple of annual salary (0.5×–2.0×) to "
            "capture recruitment, onboarding, and productivity ramp-up. "
            "Effectiveness $e$ is the biggest judgement call — the "
            "*Sensitivity* chart below sweeps it across 0–80% so you can see "
            "how much the conclusion depends on it."
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
        # Format the tier cut-offs from config so the help string stays in
        # sync with the notebook's actual band edges.
        _high = int(round(HIGH_RISK_THRESHOLD * 100))
        _mod = int(round(MODERATE_RISK_THRESHOLD * 100))
        tiers = st.multiselect(
            "Target risk tiers",
            options=["High Risk", "Moderate Risk", "Low Risk"],
            default=["High Risk", "Moderate Risk"],
            help=(
                "Which employees receive the intervention. Tiers are the "
                "fair Cox model's 1-year attrition bands: "
                f"High Risk > {_high}%, "
                f"Moderate Risk > {_mod}% up to {_high}%, "
                f"Low Risk ≤ {_mod}%. "
                "High Risk only is the most focused option; adding "
                "Moderate Risk expands coverage."
            ),
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
        # Two cost inputs deliberately phrased in parallel so the reader can
        # tell them apart at a glance:
        #   • "Intervention cost per employee"  — what we *spend* on each
        #     person targeted (flat dollars).
        #   • "Turnover cost per leaver"        — what we *lose* when someone
        #     who would have left actually leaves (× their annual salary).
        # Earlier copy used "Cost per employee" and "Replacement cost", which
        # users sometimes interpreted as the same thing in different units.
        cost_per_emp = st.number_input(
            "Intervention cost per employee ($)",
            min_value=0,
            max_value=25_000,
            value=2_000,
            step=250,
            help="What HR *spends* per person in the cohort on the retention "
            "programme — coaching, training, equity, or compensation "
            "adjustments. Flat dollar amount applied to every targeted "
            "employee, not just the ones who would have left.",
        )
        replacement_mult = st.slider(
            "Turnover cost per leaver (× annual salary)",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="What the company *loses* when an employee walks out — "
            "recruitment, onboarding, and productivity ramp-up — expressed "
            "as a multiple of their annual salary. Industry research places "
            "this at 50–200% of annual salary depending on role seniority.",
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
    break_even_cpp: float,
    cost_per_emp: float,
) -> None:
    """The main outputs: cohort size, avoided departures, net savings, ROI %,
    and the break-even cost-per-employee at the current assumptions."""
    st.subheader("Projected impact")

    avoided = expected_leavers_base - expected_leavers_post

    # Six tiles fits comfortably on a wide layout and keeps the headline
    # ROI metric in the centre, with break-even as the right-most tile so
    # it reads like the answer to "how much could I have spent?".
    c1, c2, c3, c4, c5, c6 = st.columns(6)
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
    # Delta = (current spend − break-even). Positive means the user is over-
    # spending; negative means there's headroom. ``delta_color="inverse"``
    # so that "above break-even" reads red and "below" reads green —
    # opposite of the default, since for cost more = worse.
    delta_dollar = cost_per_emp - break_even_cpp
    c6.metric(
        "Break-even cost / employee",
        f"${break_even_cpp:,.0f}",
        delta=f"{delta_dollar:+,.0f} vs. your spend",
        delta_color="inverse",
        help=(
            "Highest intervention cost per employee that still produces "
            "non-negative ROI at the current cohort size, effectiveness, "
            "and turnover-cost multiplier. Spending below this line is "
            "profitable; above it is a loss."
        ),
    )


def _render_break_even_callout(
    *,
    cost_per_emp: float,
    break_even_cpp: float,
    effectiveness: float,
) -> None:
    """Coloured plain-English summary of where the user sits vs. break-even.

    The KPI tile above shows the *number*; this banner explains the *story*
    in one sentence and gives concrete next steps when the programme is
    underwater. Three states:

      • effectiveness == 0 — no programme can ever break even (degenerate).
      • cost_per_emp ≤ break_even_cpp — green, programme makes money.
      • cost_per_emp >  break_even_cpp — yellow warning, with a remedy list.
    """
    # Streamlit renders ``st.success`` / ``st.warning`` content as markdown,
    # and markdown treats ``$...$`` as inline LaTeX. Bare dollar amounts in
    # the same line therefore get pulled into a math span, italicising the
    # text between them. Escape every literal dollar sign with a backslash
    # so it renders as a $ rather than opening a math block.
    if effectiveness <= 0 or break_even_cpp <= 0:
        # With zero effectiveness, savings = 0 and any positive spend loses
        # money — call this out explicitly rather than showing a $0 break-even
        # which reads ambiguously.
        st.warning(
            "At **0% effectiveness**, the programme cannot break even at any "
            "positive spend. Raise the *Intervention effectiveness* slider "
            "above 0% to see meaningful break-even numbers."
        )
        return

    if cost_per_emp <= break_even_cpp:
        headroom = break_even_cpp - cost_per_emp
        st.success(
            f"At these assumptions, break-even cost per employee is "
            f"approximately **\\${break_even_cpp:,.0f}**. Your spend of "
            f"**\\${cost_per_emp:,.0f}** sits **\\${headroom:,.0f} below** "
            f"that line, so the programme is in positive-ROI territory."
        )
        return

    overspend = cost_per_emp - break_even_cpp
    st.warning(
        f"At these assumptions, break-even cost per employee is approximately "
        f"**\\${break_even_cpp:,.0f}**. Your spend of **\\${cost_per_emp:,.0f}** "
        f"is **\\${overspend:,.0f} above** that line, so the programme loses "
        f"money. To turn it profitable, lower the per-employee spend, raise "
        f"effectiveness, or narrow the cohort to higher-risk employees only."
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
