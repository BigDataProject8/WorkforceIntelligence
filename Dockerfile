# syntax=docker/dockerfile:1.6
#
# Workforce Intelligence — Streamlit dashboard image.
#
# Single-stage build on Python 3.11 slim. The image is ~1.5 GB once all the
# scientific Python deps are installed (numpy, scipy, pandas, scikit-learn,
# xgboost, shap, lifelines, polars, matplotlib, streamlit).
#
# Build:
#   docker build -t workforce-intelligence .
#
# Run:
#   docker run --rm -p 8501:8501 workforce-intelligence
#
# Then open http://localhost:8501 in a browser.
#
# Note on artifacts: the dashboard reads pickled models and parquet frames
# from ./artifacts/, which is normally git-ignored. For the image to work
# you either need to commit artifacts/ to the repo (1.3 MB, harmless) or
# mount it at runtime with `-v $(pwd)/artifacts:/app/artifacts`.

# ---------------------------------------------------------------------------
# Base image — slim variant of CPython 3.11 to keep the layer count small.
# 3.11 is the floor required by numpy 2.4 / scipy 1.17 (see requirements.txt).
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# ---------------------------------------------------------------------------
# Environment hygiene.
#   PYTHONDONTWRITEBYTECODE — no .pyc clutter inside the image.
#   PYTHONUNBUFFERED        — flush stdout/stderr immediately so logs surface
#                             in `docker logs` without delay.
#   PIP_NO_CACHE_DIR        — pip's wheel cache is useless in a one-shot build
#                             and adds hundreds of MB.
#   PIP_DISABLE_PIP_VERSION_CHECK — silence the upgrade nag during builds.
# ---------------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies.
#   libgomp1 — OpenMP runtime that XGBoost needs at import time on Linux.
#              (On macOS dev machines this is supplied by `brew install libomp`.)
#   curl     — used by the HEALTHCHECK below; tiny and standard.
# Cleaning the apt cache in the same RUN keeps the layer compact.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------------------------------------------------------------------------
# Install Python dependencies first, before copying source. Doing this in a
# separate layer means edits to dashboard.py / dashboard_lib/ don't bust the
# (very expensive) pip-install cache layer on every rebuild.
# ---------------------------------------------------------------------------
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Application source.
#   dashboard.py        — Streamlit entry point.
#   dashboard_lib/      — per-tab modules.
#   artifacts/          — pickled models and parquet frames produced by the
#                         notebook's export cell. See note in the header.
# Everything else in the repo (notebook, data/raw, .git, .venv) is excluded
# by .dockerignore.
# ---------------------------------------------------------------------------
COPY dashboard.py ./
COPY dashboard_lib/ ./dashboard_lib/
COPY artifacts/ ./artifacts/

# ---------------------------------------------------------------------------
# Run as a non-root user. Streamlit doesn't need privileges and most managed
# platforms (Cloud Run, Fly, Render) reject containers that boot as root.
# ---------------------------------------------------------------------------
RUN useradd --create-home --uid 1001 streamlit \
    && chown -R streamlit:streamlit /app
USER streamlit

# Streamlit's default listen port. Document it for tooling and humans.
EXPOSE 8501

# ---------------------------------------------------------------------------
# Healthcheck — Streamlit exposes a JSON probe at /_stcore/health that
# returns "ok" once the server is fully booted. Container platforms use this
# to decide when the app is ready to receive traffic.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---------------------------------------------------------------------------
# Launch.
#   --server.address=0.0.0.0 is required for the port to be reachable from
#     outside the container; Streamlit defaults to 127.0.0.1 otherwise.
#   --server.headless=true   suppresses the "open browser" prompt and the
#     usage-stats opt-in dialog when running non-interactively.
# ---------------------------------------------------------------------------
CMD ["streamlit", "run", "dashboard.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
