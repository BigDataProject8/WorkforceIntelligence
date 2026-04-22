"""Rule-based retention action engine.

Each rule is: (trigger fn, driver feature, headline, detail). A rule surfaces
only when its trigger fires for the employee in question, and the resulting
list is ranked by that employee's SHAP contribution for the driver feature —
so the top action targets whatever is actually pushing *their* predicted
flight risk the hardest, not a generic top-N list.

The rules themselves encode domain knowledge from the notebook's Part 5
findings (overtime is the strongest modifiable driver; first 2 years are
highest-risk; satisfaction is the strongest protective factor; etc.).
"""
from __future__ import annotations

import pandas as pd


# Each tuple: (trigger callable, SHAP feature name for priority, headline, detail).
# The trigger is evaluated against a single employee row (pandas Series). We use
# ``row.get(col, default)`` so a missing column never crashes the rule loop —
# the default is set so the rule simply doesn't fire.
RECOMMENDATION_RULES: list[tuple] = [
    (
        # Overtime is the strongest modifiable risk factor in the Cox model.
        lambda r: r.get("OverTime", 0) == 1,
        "OverTime",
        "Reduce overtime load",
        "Redistribute workload across the team; overtime is the strongest "
        "modifiable risk factor in the Cox model.",
    ),
    (
        # Catches a severe single-issue pain-point even if the mean looks OK.
        lambda r: r.get("SatisfactionMin", 4) <= 1,
        "SatisfactionMin",
        "Address critical satisfaction pain-point",
        "At least one satisfaction dimension is at the lowest level. Hold a "
        "diagnostic 1:1 to identify the specific issue.",
    ),
    (
        lambda r: r.get("SatisfactionMean", 4) <= 2.5,
        "SatisfactionMean",
        "Schedule engagement conversation",
        "Composite satisfaction is below the midpoint. Career, recognition, "
        "and team-fit topics should be on the agenda.",
    ),
    (
        lambda r: r.get("YearsSinceLastPromotion", 0) >= 3,
        "YearsSinceLastPromotion",
        "Review promotion timeline",
        "Three or more years without a promotion. Confirm readiness criteria "
        "and communicate a visible path.",
    ),
    (
        # First two years are the highest-risk window per our KM curves.
        lambda r: r.get("YearsAtCompany", 99) < 2,
        "YearsAtCompany",
        "Reinforce onboarding",
        "Employee is in the first two years — the highest-risk window. Pair "
        "with a mentor and schedule 90/180-day check-ins.",
    ),
    (
        lambda r: r.get("JobInvolvement", 4) <= 2,
        "JobInvolvement",
        "Increase ownership",
        "Low job involvement. Consider a stretch assignment or a lead role "
        "on a meaningful initiative.",
    ),
    (
        lambda r: r.get("WorkLifeBalance", 4) <= 2,
        "WorkLifeBalance",
        "Offer flexibility",
        "Work-life balance rated poor. Explore remote, compressed, or "
        "staggered schedule options.",
    ),
    (
        lambda r: r.get("DistanceFromHome", 0) >= 20,
        "DistanceFromHome",
        "Hybrid / remote arrangement",
        "Commute is long enough to create daily friction. A hybrid schedule "
        "often meaningfully improves retention for this cohort.",
    ),
    (
        lambda r: r.get("TrainingTimesLastYear", 99) == 0,
        "TrainingTimesLastYear",
        "Fund training plan",
        "No training sessions in the last year. Budget a certification or "
        "course aligned with their next role.",
    ),
    (
        lambda r: r.get("StockOptionLevel", 99) == 0,
        "StockOptionLevel",
        "Review equity grant",
        "No stock options. Consider an equity refresh at the next "
        "compensation cycle.",
    ),
    (
        # Job-hopper pattern — short average tenure across prior employers.
        lambda r: r.get("AvgTenurePerCompany", 99) < 2,
        "AvgTenurePerCompany",
        "Understand mobility pattern",
        "Short average tenure across prior employers. Probe motivators and "
        "tie growth opportunities to retention milestones.",
    ),
]


def build_recommendations(
    emp_row: pd.Series,
    shap_contrib: pd.Series,
    df: pd.DataFrame,
) -> list[dict]:
    """Return triggered retention recommendations ranked by SHAP impact.

    Parameters
    ----------
    emp_row : pd.Series
        A single employee row from the processed dataframe.
    shap_contrib : pd.Series
        SHAP values for this employee, indexed by feature name. Missing
        features default to 0 priority.
    df : pd.DataFrame
        The full employee frame, used by peer-comparison rules (e.g. the
        compensation benchmark against the Job Level median).
    """
    recs: list[dict] = []

    # Evaluate the simple single-row rules.
    for trigger, feature, headline, detail in RECOMMENDATION_RULES:
        try:
            if not trigger(emp_row):
                continue
        except Exception:
            # A misbehaving rule shouldn't take down the whole panel.
            continue
        recs.append(
            {
                "headline": headline,
                "detail": detail,
                "feature": feature,
                "priority": float(shap_contrib.get(feature, 0.0)),
            }
        )

    # Peer-comparison rule — handled separately because it needs the full frame,
    # not just the row. We benchmark against the median salary within the same
    # Job Level; 85% of the median is the trigger (≈15 % below peers).
    job_level = emp_row.get("JobLevel")
    income = emp_row.get("MonthlyIncome")
    if job_level is not None and income is not None:
        peers = df.loc[df["JobLevel"] == job_level, "MonthlyIncome"]
        if len(peers) >= 10:  # Require a meaningful peer sample.
            peer_median = float(peers.median())
            if income < 0.85 * peer_median:
                gap_pct = (peer_median - income) / peer_median * 100
                recs.append(
                    {
                        "headline": "Compensation review",
                        "detail": (
                            f"Income is {gap_pct:.0f}% below the Job Level "
                            f"{int(job_level)} median (${peer_median:,.0f}). "
                            "Flag for benchmarking."
                        ),
                        "feature": "MonthlyIncome",
                        "priority": float(shap_contrib.get("MonthlyIncome", 0.0)),
                    }
                )

    # Highest SHAP push first — the top action targets the strongest driver.
    recs.sort(key=lambda r: r["priority"], reverse=True)
    return recs
