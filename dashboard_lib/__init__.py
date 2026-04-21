"""Dashboard helper package.

Each submodule owns one well-defined slice of the Streamlit app so the
``dashboard.py`` entry point stays short and the tab implementations stay
readable in isolation:

    config            — colour palette, artifact path
    artifacts         — cached loaders for pickled models and frames
    ui                — shared helpers (badges, one-hot decoding, KM plotting,
                        sidebar with glossary)
    recommendations   — rule-based retention action engine
    overview          — "Overview" tab
    risk_explorer     — "Risk Explorer" tab
    survival          — "Survival Curves" tab
    fairness          — "Fairness Audit" tab
    roi               — "ROI Calculator" tab
"""
