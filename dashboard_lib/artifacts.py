"""Cached loaders for the pickled data and model artifacts.

All loaders are wrapped in Streamlit cache decorators so heavy work (parquet
reads, joblib unpickling, SHAP explainer construction) happens at most once
per session, not once per rerun.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import polars as pl
import shap
import streamlit as st

from .config import ARTIFACTS


@st.cache_data
def load_frames() -> dict[str, pd.DataFrame]:
    """Return every DataFrame the dashboard needs.

    Fails fast with a user-friendly message if the ``artifacts/`` directory is
    missing — that's almost always because the notebook's export cell was never
    executed in the current kernel session.
    """
    if not ARTIFACTS.exists():
        st.error(
            f"Missing `{ARTIFACTS.name}/` directory. Run the final export cell "
            "in `project-08.ipynb` first."
        )
        st.stop()

    return {
        # Full processed frame (polars on disk → pandas in memory so the rest
        # of the app uses a single dataframe API).
        "df": pl.read_parquet(ARTIFACTS / "df_processed.parquet").to_pandas(),
        # Classifier-ready feature matrix and target. The columns line up with
        # the LASSO classifier's training schema — feed unscaled rows through
        # ``models["scaler"]`` before calling ``predict``/``predict_proba``.
        "X_full": pd.read_parquet(ARTIFACTS / "X_full.parquet"),
        "y_full": pd.read_parquet(ARTIFACTS / "y_full.parquet")["Attrition"],
        # Held-out 20% test split used for classifier metrics.
        "X_test": pd.read_parquet(ARTIFACTS / "X_test.parquet"),
        "y_test": pd.read_parquet(ARTIFACTS / "y_test.parquet")["Attrition"],
        # Scaled Cox input frames (already standardised — do NOT re-scale).
        "cox_fair": pd.read_parquet(ARTIFACTS / "cox_pd_fair.parquet"),
        "cox_full": pd.read_parquet(ARTIFACTS / "cox_pd_full.parquet"),
        # Per-employee Cox predictions, hazard scores, and risk tiers.
        "risk": pl.read_parquet(ARTIFACTS / "risk_scores.parquet").to_pandas(),
        # Log-rank test summary across every stratification studied.
        "logrank": pl.read_parquet(ARTIFACTS / "logrank_results.parquet").to_pandas(),
    }


@st.cache_resource
def load_models() -> dict:
    """Return the fitted models, scalers, and pre-computed SHAP array.

    Uses ``cache_resource`` instead of ``cache_data`` because these objects
    are not serialisable but are safe to share across reruns.
    """
    features = json.loads((ARTIFACTS / "cox_features.json").read_text())
    return {
        # Class-weighted LASSO Logistic Regression — selected as the dashboard's
        # primary classifier because the notebook (§3.6.2 / §3.7) flags it as
        # the most stable model and the highest-recall option (0.73 CV recall,
        # σ=0.022). For attrition, recall is the operational metric — missing a
        # leaver is far costlier than a false alarm — so this is the right
        # default for HR. The linear form also makes SHAP cheap and exact.
        "lasso": joblib.load(ARTIFACTS / "lasso_weighted.joblib"),
        # StandardScaler fitted on the training split. LASSO operates on
        # standardised features, so callers must apply this scaler to any
        # X-frame before ``predict``/``predict_proba`` or before feeding rows
        # into the SHAP LinearExplainer.
        "scaler": joblib.load(ARTIFACTS / "scaler_classifier.joblib"),
        # Fairness-constrained Cox model (no gender/age inputs).
        "cph_fair": joblib.load(ARTIFACTS / "cph_fair.joblib"),
        # Full Cox model (includes gender + age) — kept for comparison.
        "cph_full": joblib.load(ARTIFACTS / "cph_full.joblib"),
        "cox_features": features,
        # Raw survival arrays — useful for refitting the overall KM curve
        # without reloading the full dataframe.
        "durations": np.load(ARTIFACTS / "durations.npy"),
        "events": np.load(ARTIFACTS / "events.npy"),
        # Pre-computed SHAP values from the LinearExplainer over the scaled
        # test split. Same shape (n_test, n_features) as the previous XGBoost
        # array, so any consumer that just needs mean-|SHAP| works unchanged.
        "shap_test": np.load(ARTIFACTS / "shap_values_lasso_w.npy"),
    }


@st.cache_resource
def build_shap_explainer(_model, _background) -> shap.LinearExplainer:
    """Build a LinearExplainer once per session.

    The leading underscores tell Streamlit *not* to hash the arguments —
    sklearn estimators and large numpy arrays are expensive (or impossible)
    to hash, and the explainer only depends on the fitted coefficients +
    background distribution, both stable for the session.

    Parameters
    ----------
    _model : sklearn LogisticRegression
        The fitted LASSO classifier.
    _background : np.ndarray
        The standardised feature matrix used as the SHAP background
        distribution. Pass ``scaler.transform(X)`` once per session.
    """
    return shap.LinearExplainer(_model, _background)
