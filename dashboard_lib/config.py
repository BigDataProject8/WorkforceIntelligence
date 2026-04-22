"""Constants shared across the dashboard — palette, artifact location, tier
thresholds. Kept in one place so styling stays consistent if we rebrand later.
"""
from __future__ import annotations

from pathlib import Path

# Location of pickled artifacts produced by the notebook's final export cell.
# Resolved relative to the repo root (i.e. the package's parent) so the app
# works no matter where Streamlit is invoked from.
ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"

# Sequential palette matching the notebook's figures so charts look consistent
# between the written report and the live app.
PALETTE = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]

# Risk-tier colour mapping. Red = High (act now), amber = Moderate (monitor),
# green = Low (standard engagement).
TIER_COLORS = {
    "High Risk": "#F44336",
    "Moderate Risk": "#FF9800",
    "Low Risk": "#4CAF50",
}

# Probability cut-offs used by the notebook when assigning risk tiers.
# Duplicated here so the ROI calculator can describe them in plain English.
HIGH_RISK_THRESHOLD = 0.35
MODERATE_RISK_THRESHOLD = 0.15

# Fairness thresholds taken from the notebook's audit section.
DISPARITY_FLAG = 1.25
CALIBRATION_GAP_FLAG_PP = 10.0
C_INDEX_GAP_FLAG = 0.05
