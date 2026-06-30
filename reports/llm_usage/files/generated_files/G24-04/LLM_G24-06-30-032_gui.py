"""
gui.py — Streamlit front-end for the QUBO Feature Selection & Classification Pipeline.

Launch with:
    streamlit run src/qubo_project/gui.py
or:
    python -m qubo_project.gui

─────────────────────────────────────────────────────────────────────────────
PART 1 of 2 — Implements:
  • Sidebar  : dataset upload, pipeline status tracker, global controls
  • Phase 1  : Preprocessing  (fit_normalize)
  • Phase 2  : Feature Selection  (select_features + plotly alpha chart)
  • Phase 3/4: Placeholder panels with phase-gate banners
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# ── PATH SETUP ────────────────────────────────────────────────────────────────
# Allows `streamlit run src/qubo_project/gui.py` from the project root.
_HERE = Path(__file__).resolve().parent   # src/qubo_project/
_SRC  = _HERE.parent                      # src/
for _p in [str(_SRC), str(_SRC.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── BACKEND IMPORTS ───────────────────────────────────────────────────────────
_BACKEND_OK  = True
_BACKEND_ERR = ""
try:
    from qubo_project.preprocessing import fit_normalize
    from qubo_project.feature_selection import select_features
    from qubo_project.model import train, predict, CLASSIFIERS
except ImportError as _exc:
    _BACKEND_OK  = False
    _BACKEND_ERR = str(_exc)

# ── PAGE CONFIG  (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="QUBO Classifier Pipeline",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — injected as a single <style> block
# Theme:  deep-charcoal background · electric-teal accent
# Signature element: pipeline status tracker with animated step badges
# ═════════════════════════════════════════════════════════════════════════════
_CSS = """
<style>

/* ═══════════════════════════════════════════════════════════════
   THEME TOKENS
   All colour decisions live here as CSS custom properties.
   prefers-color-scheme lets the GUI adapt automatically to the
   user's system preference — no forced dark backgrounds.
   ═══════════════════════════════════════════════════════════════ */

/* ── DARK MODE (default) ──────────────────────────────────────── */
:root {
    /* Accent — electric teal */
    --accent:       #00C9A7;
    --accent-glow:  rgba(0,201,167,.16);
    --accent-edge:  rgba(0,201,167,.30);

    /* Surfaces — semi-transparent so they float above any bg */
    --card-bg:      rgba(255,255,255,.045);
    --card-border:  rgba(255,255,255,.09);

    /* Typography */
    --text-primary: #E6EDF3;
    --text-muted:   #8B949E;

    /* Tab bar */
    --tab-bar-bg:   rgba(255,255,255,.04);
    --tab-color:    rgba(255,255,255,.48);

    /* Status — ok */
    --ok-clr:       #3FB950;
    --ok-bg:        rgba(63,185,80,.11);

    /* Status — warning */
    --warn-clr:     #F0883E;
    --warn-bg:      rgba(240,136,62,.09);
    --warn-bd:      rgba(240,136,62,.28);

    /* Status — info */
    --blue-clr:     #58A6FF;
    --info-bg:      rgba(88,166,255,.08);
    --info-bd:      rgba(88,166,255,.22);

    /* Status — error */
    --err-clr:      #FF6B6B;

    /* Pipeline step states */
    --ps-done-bg:         rgba(63,185,80,.10);
    --ps-pending-clr:     rgba(255,255,255,.38);
    --ps-pending-badge:   rgba(255,255,255,.10);

    /* Text colour placed ON TOP of var(--accent).
       Dark-mode accent (#00C9A7) is bright → black reads best. */
    --accent-contrast: #000;

    --radius: 10px;
    --mono:   "JetBrains Mono","Fira Code",ui-monospace,monospace;
}

/* ── LIGHT MODE ───────────────────────────────────────────────── */
@media (prefers-color-scheme: light) {
    :root {
        /* Teal shifted darker for contrast on white */
        --accent:       #0A8A74;
        --accent-glow:  rgba(10,138,116,.12);
        --accent-edge:  rgba(10,138,116,.25);

        /* Surfaces — very subtle dark tint on white */
        --card-bg:      rgba(0,0,0,.03);
        --card-border:  rgba(0,0,0,.10);

        /* Typography */
        --text-primary: #1F2328;
        --text-muted:   #656D76;

        /* Tab bar */
        --tab-bar-bg:   rgba(0,0,0,.04);
        --tab-color:    rgba(0,0,0,.45);

        /* Status — ok */
        --ok-clr:       #1A7F37;
        --ok-bg:        rgba(26,127,55,.08);

        /* Status — warning */
        --warn-clr:     #9A6700;
        --warn-bg:      rgba(154,103,0,.07);
        --warn-bd:      rgba(154,103,0,.25);

        /* Status — info */
        --blue-clr:     #0969DA;
        --info-bg:      rgba(9,105,218,.07);
        --info-bd:      rgba(9,105,218,.20);

        /* Status — error */
        --err-clr:      #CF222E;

        /* Pipeline step states */
        --ps-done-bg:         rgba(26,127,55,.08);
        --ps-pending-clr:     rgba(0,0,0,.35);
        --ps-pending-badge:   rgba(0,0,0,.08);

        /* Light-mode accent (#0A8A74) is a darkened teal, sized for
           text-on-white contrast — it is NOT light, so it needs a
           light label on top of it, not a dark one. */
        --accent-contrast: #fff;
    }
}

/* ── GLOBAL ─────────────────────────────────────────────────────── */
.main .block-container {
    padding-top: 1.4rem !important;
    max-width: 1300px;
}

/* ── SIDEBAR — Streamlit controls the background; we just add a
   divider line so the panel boundary remains visible. ─────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid var(--card-border);
}

/* ── TABS ────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--tab-bar-bg) !important;
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--card-border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    padding: .4rem 1rem !important;
    color: var(--tab-color) !important;
    font-size: .87rem;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-glow) !important;
    color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.25rem !important;
}

/* ── BUTTONS ─────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: .01em;
    transition: transform .15s, box-shadow .15s !important;
}
.stButton > button:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px var(--accent-glow) !important;
}

/* ── PAGE HEADER ─────────────────────────────────────────────────── */
.page-title {
    font-size: 1.55rem; font-weight: 800;
    letter-spacing: -.03em; margin: 0;
    color: var(--text-primary);
}
.page-subtitle {
    color: var(--text-muted); font-size: .88rem; margin: .25rem 0 0;
}

/* ── PHASE HEADER ────────────────────────────────────────────────── */
.phase-header {
    display: flex;
    align-items: center;
    gap: .85rem;
    margin-bottom: 1.4rem;
    padding-bottom: .85rem;
    border-bottom: 1px solid var(--card-border);
}
.phase-number {
    background: var(--accent);
    color: var(--accent-contrast);
    width: 38px; height: 38px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 1.05rem;
    flex-shrink: 0;
    box-shadow: 0 0 16px var(--accent-glow);
}
.phase-title    { font-size: 1.22rem; font-weight: 700; color: var(--text-primary); margin: 0; }
.phase-subtitle { font-size: .82rem;  color: var(--text-muted);                     margin: 0; }

/* ── PARAM CARD ──────────────────────────────────────────────────── */
.param-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.15rem 1.4rem 1.25rem;
    margin-bottom: 1.1rem;
}
.param-card h4 { margin: 0 0 .9rem; font-size: 1rem; color: var(--text-primary); }

/* ── METRIC GRID ─────────────────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
    gap: .7rem;
    margin: 1rem 0 1.15rem;
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: .85rem 1rem .9rem;
    text-align: center;
}
.metric-card .mc-label {
    font-size: .7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: .07em;
    margin-bottom: .3rem;
}
.metric-card .mc-value {
    font-size: 1.42rem; font-weight: 700;
    color: var(--accent);
    font-family: var(--mono);
    line-height: 1.15;
}
.metric-card .mc-unit { font-size: .7rem; color: var(--text-muted); margin-top: .15rem; }

/* ── PIPELINE STATUS TRACKER ─────────────────────────────────────── */
.pipeline-step {
    display: flex; align-items: center; gap: .7rem;
    padding: .55rem .7rem;
    border-radius: 8px;
    margin-bottom: .35rem;
    font-size: .86rem; font-weight: 500;
}
.pipeline-step.ps-done    { background: var(--ps-done-bg);  color: var(--ok-clr);  }
.pipeline-step.ps-active  {
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid var(--accent-edge);
}
.pipeline-step.ps-pending { background: transparent; color: var(--ps-pending-clr); }

.ps-badge {
    width: 26px; height: 26px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .76rem; font-weight: 700; flex-shrink: 0;
}
.ps-done    .ps-badge { background: var(--ok-clr);           color: #fff; }
.ps-active  .ps-badge { background: var(--accent);           color: var(--accent-contrast); }
.ps-pending .ps-badge { background: var(--ps-pending-badge); color: var(--ps-pending-clr); }

/* ── LOCK BANNER ─────────────────────────────────────────────────── */
.lock-banner {
    display: flex; align-items: center; gap: .7rem;
    background: var(--warn-bg);
    border: 1px solid var(--warn-bd);
    border-radius: var(--radius);
    padding: .95rem 1.2rem;
    color: var(--warn-clr);
    font-size: .9rem;
    margin-bottom: 1rem;
}

/* ── FEATURE CHIPS ───────────────────────────────────────────────── */
.chip-list { display: flex; flex-wrap: wrap; gap: .38rem; margin-top: .5rem; }
.chip {
    background: var(--accent-glow);
    border: 1px solid var(--accent-edge);
    color: var(--accent);
    border-radius: 20px;
    padding: .18rem .6rem;
    font-size: .74rem;
    font-family: var(--mono);
}

/* ── INFO BOX ────────────────────────────────────────────────────── */
.info-box {
    background: var(--info-bg);
    border: 1px solid var(--info-bd);
    border-radius: 8px;
    padding: .75rem 1rem;
    color: var(--blue-clr);
    font-size: .88rem;
    margin: .75rem 0;
}

/* ── SIDEBAR LOGO ────────────────────────────────────────────────── */
.sidebar-logo {
    text-align: center;
    padding: .4rem 0 1.1rem;
    border-bottom: 1px solid var(--card-border);
    margin-bottom: 1rem;
}
.sidebar-logo .logo-icon  { font-size: 2.2rem; line-height: 1; }
.sidebar-logo .logo-title {
    font-weight: 800; font-size: 1.1rem;
    color: var(--text-primary);
    letter-spacing: -.02em; margin-top: .3rem;
}
.sidebar-logo .logo-sub { font-size: .73rem; color: var(--text-muted); margin-top: .15rem; }

/* ── SECTION LABEL ───────────────────────────────────────────────── */
.section-label {
    font-size: .72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .09em;
    color: var(--text-muted);
    margin: .85rem 0 .4rem;
}

</style>
"""


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE — initialise all keys exactly once per session
# ═════════════════════════════════════════════════════════════════════════════
_STATE_DEFAULTS: dict = {
    # Dataset
    "dataset_path":    None,   # str: absolute path to saved temp CSV
    "dataset_name":    None,   # str: original filename shown to user
    "dataset_preview": None,   # DataFrame: first 5 rows for the sidebar preview
    "target_column":   "target",

    # Phase 1 — Preprocessing
    "preprocessing_done":  False,
    "preprocessing_stats": None,   # dict returned by fit_normalize
    "normalized_csv":      None,   # str: path to outputs/normalized.csv

    # Phase 2 — Feature Selection
    "feature_selection_done":  False,
    "feature_selection_stats": None,
    "reduced_train_csv":       None,
    "reduced_test_csv":        None,

    # Phase 3 — Training  (populated in Part 2)
    "training_done":  False,
    "training_stats": None,
    "model_path":     None,

    # Phase 4 — Prediction  (populated in Part 2)
    "prediction_done":  False,
    "prediction_stats": None,
}


def _init_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ═════════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _outputs_dir() -> Path:
    """Mirror the backend's _resolve_outputs_dir() logic so GUI paths match."""
    here      = Path(__file__).resolve().parent
    candidate = here
    for _ in range(6):
        outputs = candidate / "outputs"
        if outputs.is_dir():
            return outputs
        candidate = candidate.parent
    fallback = Path.cwd() / "outputs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _save_upload(uploaded_file) -> Path:
    """Persist an UploadedFile object to a stable temp location and return the path."""
    tmp_dir = Path(tempfile.gettempdir()) / "qubo_gui_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / uploaded_file.name
    uploaded_file.seek(0)
    dest.write_bytes(uploaded_file.read())
    return dest


# ═════════════════════════════════════════════════════════════════════════════
# HTML FRAGMENT HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _html_phase_header(number: str, title: str, subtitle: str) -> str:
    return (
        f'<div class="phase-header">'
        f'  <div class="phase-number">{number}</div>'
        f'  <div>'
        f'    <div class="phase-title">{title}</div>'
        f'    <div class="phase-subtitle">{subtitle}</div>'
        f'  </div>'
        f'</div>'
    )


def _html_lock_banner(reason: str) -> str:
    return (
        f'<div class="lock-banner">'
        f'🔒 &nbsp;<strong>Phase locked.</strong>&nbsp; {reason}'
        f'</div>'
    )


def _html_metric_card(label: str, value, unit: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'  <div class="mc-label">{label}</div>'
        f'  <div class="mc-value">{value}</div>'
        f'  <div class="mc-unit">{unit}</div>'
        f'</div>'
    )


def _html_metric_grid(*cards: str) -> str:
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def _html_chip_list(names: list[str]) -> str:
    chips = "".join(f'<span class="chip">{n}</span>' for n in names)
    return f'<div class="chip-list">{chips}</div>'


def _html_pipeline_status() -> str:
    phases = [
        ("Preprocessing",     "preprocessing_done"),
        ("Feature Selection", "feature_selection_done"),
        ("Training",          "training_done"),
        ("Prediction",        "prediction_done"),
    ]
    done_count = sum(1 for _, k in phases if st.session_state.get(k, False))
    active_idx = done_count  # index of the next incomplete phase

    rows: list[str] = []
    for i, (label, key) in enumerate(phases):
        is_done   = bool(st.session_state.get(key, False))
        is_active = (i == active_idx) and not is_done
        state     = "ps-done" if is_done else ("ps-active" if is_active else "ps-pending")
        badge     = "✓" if is_done else str(i + 1)
        rows.append(
            f'<div class="pipeline-step {state}">'
            f'  <div class="ps-badge">{badge}</div>'
            f'  <span>Phase {i + 1} &nbsp;·&nbsp; {label}</span>'
            f'</div>'
        )
    return "\n".join(rows)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:

        # ── Logo / branding ────────────────────────────────────────────────
        st.markdown(
            '<div class="sidebar-logo">'
            '  <div class="logo-icon">⚛️</div>'
            '  <div class="logo-title">QUBO Classifier</div>'
            '  <div class="logo-sub">Feature Reduction · Binary Classification</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # ── Backend health check ───────────────────────────────────────────
        if not _BACKEND_OK:
            st.error(f"⚠️ Backend import failed:\n`{_BACKEND_ERR}`")
            st.caption("Check that all project modules are in `src/qubo_project/`.")

        # ── Dataset loader ─────────────────────────────────────────────────
        st.markdown('<div class="section-label">📂 Dataset</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            help=(
                "Upload the input CSV. "
                "The first row must contain column headers; "
                "all feature columns must be numeric."
            ),
            label_visibility="collapsed",
        )

        if uploaded is not None:
            _handle_upload(uploaded)

        # Dataset info chips
        if st.session_state.dataset_path and st.session_state.dataset_preview is not None:
            preview: pd.DataFrame = st.session_state.dataset_preview
            n_cols = len(preview.columns)
            st.success(f"✓ **{st.session_state.dataset_name}** — {n_cols} columns detected")

            with st.expander("Preview (first 5 rows)", expanded=False):
                st.dataframe(preview, use_container_width=True, height=170)

        # ── Global: target column (shared across phases) ───────────────────
        st.markdown('<div class="section-label">🎯 Target Column</div>', unsafe_allow_html=True)
        new_target = st.text_input(
            "Name of the binary (0 / 1) target column",
            value=st.session_state.target_column,
            label_visibility="collapsed",
            placeholder="e.g.  target",
            help="This name is passed to every pipeline phase automatically.",
        )
        if new_target.strip():
            st.session_state.target_column = new_target.strip()
        else:
            st.warning("Target column name cannot be empty.")

        st.divider()

        # ── Pipeline status tracker ─────────────────────────────────────────
        st.markdown('<div class="section-label">🔬 Pipeline Status</div>', unsafe_allow_html=True)
        st.markdown(_html_pipeline_status(), unsafe_allow_html=True)

        st.divider()

        # ── Reset ──────────────────────────────────────────────────────────
        if st.button("↺  Reset Pipeline", use_container_width=True):
            # Wipe all managed state keys
            for key in list(_STATE_DEFAULTS.keys()):
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        st.caption("QUBO Classification Project · v1.0")


def _handle_upload(uploaded_file) -> None:
    """Validate the upload, persist it, and update session state."""
    try:
        # Peek to validate structure
        peek = pd.read_csv(uploaded_file, nrows=5)

        if peek.empty:
            st.error("The uploaded file is empty.")
            return
        if peek.shape[1] == 0:
            st.error("No columns were detected. Check the CSV format.")
            return

        # Persist to disk so the backend can read it by path
        uploaded_file.seek(0)
        dest = _save_upload(uploaded_file)

        st.session_state.dataset_path    = str(dest)
        st.session_state.dataset_name    = uploaded_file.name
        st.session_state.dataset_preview = peek

        # Uploading a new file resets all downstream phases
        st.session_state.preprocessing_done  = False
        st.session_state.preprocessing_stats = None
        st.session_state.normalized_csv      = None
        st.session_state.feature_selection_done  = False
        st.session_state.feature_selection_stats = None
        st.session_state.training_done  = False
        st.session_state.prediction_done = False

    except Exception as exc:
        st.error(f"Could not read the uploaded file: {exc}")
        st.session_state.dataset_path = None


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 1 — PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def _tab_preprocessing() -> None:
    st.markdown(
        _html_phase_header(
            "1", "Preprocessing",
            "Drop sparse columns, impute missing values, and apply Z-score normalisation."
        ),
        unsafe_allow_html=True,
    )

    # ── Phase gate ─────────────────────────────────────────────────────────
    if not st.session_state.dataset_path:
        st.markdown(
            _html_lock_banner("Upload a CSV dataset in the sidebar to unlock this phase."),
            unsafe_allow_html=True,
        )
        return

    # ── Parameter card ─────────────────────────────────────────────────────
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    st.markdown("<h4>⚙️ Parameters</h4>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        target_col_display = st.session_state.target_column
        st.markdown(
            f'<div class="info-box">'
            f'Target column set in sidebar: &nbsp;<strong><code>{target_col_display}</code></strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption("Change it via the **Target Column** field in the sidebar.")

    with col2:
        min_perc_valid = st.slider(
            "Min. valid-data fraction per column",
            min_value=0.01, max_value=1.00,
            value=0.05, step=0.01, format="%.2f",
            help=(
                "Fraction of rows that must be non-NaN AND non-zero for a column "
                "to survive. E.g. 0.05 keeps columns with ≥ 5 % valid data."
            ),
        )
        threshold_pct = f"{min_perc_valid * 100:.0f}%"
        st.caption(
            f"Columns with fewer than **{threshold_pct}** valid rows will be dropped."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Inline validation ───────────────────────────────────────────────────
    warnings_list: list[str] = []

    target = st.session_state.target_column
    preview = st.session_state.dataset_preview

    if not target:
        warnings_list.append("Target column name is empty — set it in the sidebar.")
    elif preview is not None and target not in preview.columns:
        sample = ", ".join(f"`{c}`" for c in list(preview.columns)[:8])
        warnings_list.append(
            f"Column **`{target}`** not found in the preview. "
            f"Detected columns include: {sample} …"
        )

    for w in warnings_list:
        st.warning(w)

    # ── Run ────────────────────────────────────────────────────────────────
    run_disabled = bool(warnings_list) or not _BACKEND_OK

    if st.button(
        "▶  Start Preprocessing",
        type="primary",
        disabled=run_disabled,
        key="btn_preprocess",
    ):
        _run_preprocessing(target, min_perc_valid)

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.preprocessing_done and st.session_state.preprocessing_stats:
        _render_preprocessing_results(st.session_state.preprocessing_stats)


def _run_preprocessing(target_column: str, min_perc_valid: float) -> None:
    with st.spinner("Running preprocessing — this may take a moment for large datasets …"):
        try:
            stats = fit_normalize(
                input_csv         = st.session_state.dataset_path,
                target_column     = target_column,
                normalized_csv    = "normalized.csv",
                outInitalRes_json = "preprocessing_result.json",
                minPercValid      = min_perc_valid,
            )

            st.session_state.preprocessing_done  = True
            st.session_state.preprocessing_stats = stats
            st.session_state.normalized_csv = str(_outputs_dir() / "normalized.csv")

            # Invalidate all downstream phases
            st.session_state.feature_selection_done  = False
            st.session_state.feature_selection_stats = None
            st.session_state.reduced_train_csv        = None
            st.session_state.reduced_test_csv         = None
            st.session_state.training_done   = False
            st.session_state.prediction_done = False

        except FileNotFoundError as exc:
            st.error(f"**File not found:** {exc}")
        except ValueError as exc:
            st.error(f"**Validation error:** {exc}")
        except Exception as exc:
            st.error(f"**Unexpected error:** {exc}")


def _render_preprocessing_results(stats: dict) -> None:
    st.success("✓ Preprocessing complete — normalised dataset saved to `outputs/`.")

    dropped_count = len(stats.get("dropped_feature_names", []))

    # Metric grid
    grid_html = _html_metric_grid(
        _html_metric_card("Input Features",   stats.get("n_input_features", "—")),
        _html_metric_card("Kept Features",    stats.get("n_kept_features",  "—")),
        _html_metric_card("Dropped Columns",  dropped_count),
        _html_metric_card("Dataset Rows",     f"{stats.get('dataset_size', 0):,}", "rows"),
        _html_metric_card("Read Time",        f"{stats.get('dataset_input_time', 0):.2f}", "s"),
        _html_metric_card("Process Time",     f"{stats.get('dataset_processing_time', 0):.2f}", "s"),
    )
    st.markdown(grid_html, unsafe_allow_html=True)

    # Dropped column chips
    dropped_names: list[str] = stats.get("dropped_feature_names", [])
    if dropped_names:
        with st.expander(f"🗑 Dropped columns ({len(dropped_names)})", expanded=False):
            st.markdown(_html_chip_list(dropped_names), unsafe_allow_html=True)
            st.caption(
                "These columns contained too few valid (non-NaN, non-zero) rows "
                "and were removed before normalisation."
            )

    # Full JSON
    with st.expander("📄 Full statistics (JSON)", expanded=False):
        st.json(stats)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2 — FEATURE SELECTION (QUBO)
# ═════════════════════════════════════════════════════════════════════════════

def _tab_feature_selection() -> None:
    st.markdown(
        _html_phase_header(
            "2", "Feature Selection",
            "Build and solve a QUBO problem to find the most informative, "
            "non-redundant features via Simulated Annealing."
        ),
        unsafe_allow_html=True,
    )

    # ── Phase gate ─────────────────────────────────────────────────────────
    if not st.session_state.preprocessing_done:
        st.markdown(
            _html_lock_banner(
                "Complete <strong>Phase 1 — Preprocessing</strong> before running "
                "feature selection."
            ),
            unsafe_allow_html=True,
        )
        return

    n_kept = (st.session_state.preprocessing_stats or {}).get("n_kept_features", 0)

    # ── Parameter card ─────────────────────────────────────────────────────
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    st.markdown("<h4>⚙️ QUBO Parameters</h4>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        perc_selected = st.slider(
            "Fraction of features to select (percSelected)",
            min_value=0.05, max_value=0.80, value=0.20, step=0.01, format="%.2f",
            help="Target proportion of features to retain. 0.20 → keep ~20 % of features.",
        )
        perc_test = st.slider(
            "Test-set fraction (percTest)",
            min_value=0.10, max_value=0.50, value=0.30, step=0.05, format="%.2f",
            help="Fraction of the dataset reserved for the test set.",
        )

    with col_b:
        allowance = int(st.number_input(
            "Feature allowance  (±N features)",
            min_value=0, max_value=20, value=1, step=1,
            help=(
                "Tolerance around the target feature count K. "
                "The search stops when K−allowance ≤ selected ≤ K+allowance."
            ),
        ))
        alpha_computations = int(st.number_input(
            "Max α iterations (alpha_computations)",
            min_value=5, max_value=500, value=100, step=5,
            help="Maximum number of bisection iterations when searching for the right α.",
        ))

    seed = int(st.number_input(
        "Random seed",
        min_value=0, max_value=2**31 - 1, value=42, step=1,
        help="Seed for the Simulated Annealing solver — ensures reproducible results.",
    ))

    # Target K preview
    if n_kept > 0:
        K = round(perc_selected * n_kept)
        lo = max(0, K - allowance)
        hi = K + allowance
        st.markdown(
            f'<div class="info-box">'
            f'With <strong>{n_kept}</strong> features after preprocessing, '
            f'target&nbsp;K&nbsp;= round({perc_selected}&nbsp;×&nbsp;{n_kept})&nbsp;'
            f'= <strong>{K}</strong>&nbsp; → acceptable range&nbsp;'
            f'<strong>[{lo}, {hi}]</strong> features.'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close param-card

    # ── Run ────────────────────────────────────────────────────────────────
    if not _BACKEND_OK:
        st.error(f"Backend import error — cannot run: `{_BACKEND_ERR}`")
        return

    if st.button("▶  Start Feature Selection", type="primary", key="btn_feat_sel"):
        _run_feature_selection(
            perc_selected      = perc_selected,
            perc_test          = perc_test,
            allowance          = allowance,
            alpha_computations = alpha_computations,
            seed               = seed,
        )

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.feature_selection_done and st.session_state.feature_selection_stats:
        _render_feature_selection_results(st.session_state.feature_selection_stats)


def _run_feature_selection(
    perc_selected:      float,
    perc_test:          float,
    allowance:          int,
    alpha_computations: int,
    seed:               int,
) -> None:
    progress = st.progress(0, text="Initialising QUBO solver …")
    try:
        with st.spinner(
            "Running QUBO optimisation — may take several minutes for large feature sets …"
        ):
            progress.progress(10, text="Computing Spearman correlations …")
            select_features(
                normalized_csv    = st.session_state.normalized_csv,
                reducedTrain_csv  = "training_reduced.csv",
                reducedTest_csv   = "test_reduced.csv",
                output_ottim_csv  = "optimizations.csv",
                output_json       = "feature_selection_result.json",
                target_column     = st.session_state.target_column,
                percTest          = perc_test,
                percSelected      = perc_selected,
                allowance         = allowance,
                seed              = seed,
                alpha_computations= alpha_computations,
            )
            progress.progress(90, text="Loading results …")

        # Load JSON produced by backend
        json_path = _outputs_dir() / "feature_selection_result.json"
        with open(json_path, encoding="utf-8") as fh:
            fs_stats = json.load(fh)

        st.session_state.feature_selection_done  = True
        st.session_state.feature_selection_stats = fs_stats
        st.session_state.reduced_train_csv = str(_outputs_dir() / "training_reduced.csv")
        st.session_state.reduced_test_csv  = str(_outputs_dir() / "test_reduced.csv")

        # Invalidate downstream phases
        st.session_state.training_done   = False
        st.session_state.prediction_done = False

        progress.progress(100, text="Optimisation complete ✓")

    except FileNotFoundError as exc:
        progress.empty()
        st.error(f"**File not found:** {exc}")
    except ValueError as exc:
        progress.empty()
        st.error(f"**Validation error:** {exc}")
    except Exception as exc:
        progress.empty()
        st.error(f"**Unexpected error during QUBO optimisation:** {exc}")


def _render_feature_selection_results(fs: dict) -> None:
    st.success(
        f"✓ Feature selection complete — "
        f"**{fs.get('n_selected', '?')}** features selected "
        f"(α = {fs.get('alpha', 0):.4f})."
    )

    # Metric grid
    grid_html = _html_metric_grid(
        _html_metric_card("Total Features",   fs.get("n_features",   "—")),
        _html_metric_card("Target K",         fs.get("target_k",     "—")),
        _html_metric_card("Selected",         fs.get("n_selected",   "—")),
        _html_metric_card("Allowance",        f"±{fs.get('allowance', '—')}"),
        _html_metric_card("Best α",           f"{fs.get('alpha', 0):.4f}"),
        _html_metric_card("α Iterations",     fs.get("alpha_computations", "—")),
        _html_metric_card("Q-matrix Time",    f"{fs.get('q_matrix_creation_time', 0):.2f}", "s"),
        _html_metric_card("Mean Optim. Time", f"{fs.get('mean_optimization_time', 0):.3f}", "s"),
        _html_metric_card("Std Optim. Time",  f"{fs.get('std_dev_optimization_time', 0):.3f}", "s"),
        _html_metric_card("Training Rows",    f"{fs.get('training_dataset_size', 0):,}"),
        _html_metric_card("Test Rows",        f"{fs.get('test_dataset_size', 0):,}"),
    )
    st.markdown(grid_html, unsafe_allow_html=True)

    # Selected feature chips
    sel_names: list[str] = fs.get("selected_feature_names", [])
    if sel_names:
        st.markdown(
            f"**Selected features ({len(sel_names)}):**",
            help="These columns survived QUBO optimisation and are used for training.",
        )
        st.markdown(_html_chip_list(sel_names), unsafe_allow_html=True)
        st.write("")  # spacing

    # ── Alpha search chart ────────────────────────────────────────────────
    ottim_path = _outputs_dir() / "optimizations.csv"
    if ottim_path.exists():
        with st.expander("📈 Alpha search trajectory", expanded=True):
            _render_alpha_chart(ottim_path, fs)

    # ── Raw JSON ──────────────────────────────────────────────────────────
    with st.expander("📄 Full statistics (JSON)", expanded=False):
        st.json(fs)


def _detect_theme_base() -> str:
    """Best-effort detection of the active Streamlit theme ('light' or 'dark').

    Tries the modern `st.context.theme` API first (reflects the resolved
    theme even when the user has "Use system setting" selected), then
    falls back to the static `theme.base` config option, then to 'dark'
    (the original design's default) if neither is available.
    """
    try:
        theme_type = st.context.theme.type  # Streamlit ≥ 1.36
        if theme_type in ("light", "dark"):
            return theme_type
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("light", "dark"):
            return base
    except Exception:
        pass
    return "dark"


# Chart palettes mirror the CSS tokens in _CSS exactly, so the Plotly
# figure always matches the surrounding page instead of fighting it.
_CHART_PALETTES: dict = {
    "dark": dict(
        template     = "plotly_dark",
        paper_bg     = "rgba(0,0,0,0)",
        plot_bg      = "rgba(255,255,255,.03)",
        grid         = "#30363D",
        text         = "#8B949E",
        accent       = "#00C9A7",
        blue         = "#58A6FF",
        marker_edge  = "#0D1117",
        band_fill    = "rgba(88,166,255,.12)",
    ),
    "light": dict(
        template     = "plotly_white",
        paper_bg     = "rgba(0,0,0,0)",
        plot_bg      = "rgba(0,0,0,.015)",
        grid         = "#D8DEE4",
        text         = "#656D76",
        accent       = "#0A8A74",
        blue         = "#0969DA",
        marker_edge  = "#FFFFFF",
        band_fill    = "rgba(9,105,218,.08)",
    ),
}


def _render_alpha_chart(ottim_path: Path, fs: dict) -> None:
    """Render the α-vs-features chart using Plotly (falls back to st.dataframe)."""
    try:
        import plotly.graph_objects as go

        pal = _CHART_PALETTES[_detect_theme_base()]

        df_opt = pd.read_csv(ottim_path).sort_values("alpha")
        target_k  = fs.get("target_k")
        allowance = fs.get("allowance", 1)
        best_alpha = fs.get("alpha")

        fig = go.Figure()

        # Main line: α → n_selected
        fig.add_trace(go.Scatter(
            x    = df_opt["alpha"],
            y    = df_opt["n_selected"],
            mode = "lines+markers",
            name = "Features selected",
            line = dict(color=pal["accent"], width=2.5),
            marker= dict(size=7, color=pal["accent"],
                         line=dict(width=1.5, color=pal["marker_edge"])),
        ))

        # Cost line on secondary y-axis
        fig.add_trace(go.Scatter(
            x    = df_opt["alpha"],
            y    = df_opt["cost_value"],
            mode = "lines",
            name = "QUBO cost",
            line = dict(color=pal["blue"], width=1.5, dash="dot"),
            yaxis= "y2",
            opacity=0.75,
        ))

        # Acceptable band for target K
        if target_k is not None:
            fig.add_hrect(
                y0=max(0, target_k - allowance),
                y1=target_k + allowance,
                fillcolor=pal["band_fill"],
                line_width=0,
                annotation_text=f"K ± {allowance}  ({target_k - allowance}–{target_k + allowance})",
                annotation_font_color=pal["blue"],
                annotation_position="right",
            )

        # Best alpha vertical marker
        if best_alpha is not None:
            fig.add_vline(
                x=best_alpha,
                line_width=1.5,
                line_dash="dash",
                line_color=pal["accent"],
                annotation_text=f"α* = {best_alpha:.4f}",
                annotation_font_color=pal["accent"],
                annotation_position="top left",
            )

        fig.update_layout(
            template     = pal["template"],
            paper_bgcolor= pal["paper_bg"],
            plot_bgcolor = pal["plot_bg"],
            xaxis_title  = "α  (weight of influence vs independence)",
            yaxis_title  = "Features selected",
            yaxis2       = dict(
                title      = "QUBO cost",
                overlaying = "y",
                side       = "right",
                showgrid   = False,
                color      = pal["blue"],
            ),
            height       = 340,
            margin       = dict(l=10, r=10, t=24, b=10),
            legend       = dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            font         = dict(size=12, color=pal["text"]),
            xaxis        = dict(
                gridcolor=pal["grid"], zerolinecolor=pal["grid"],
            ),
            yaxis        = dict(
                gridcolor=pal["grid"], zerolinecolor=pal["grid"],
            ),
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The shaded band marks the acceptable feature count. "
            "The dashed line shows the best α found by bisection search."
        )

    except ImportError:
        # Graceful fallback if plotly is absent
        st.dataframe(
            pd.read_csv(ottim_path).sort_values("alpha"),
            use_container_width=True,
        )
        st.caption("Install `plotly` for an interactive chart.")

    except Exception as exc:
        st.warning(f"Chart rendering failed: {exc}")
        st.dataframe(
            pd.read_csv(ottim_path).sort_values("alpha"),
            use_container_width=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3 — TRAINING  (placeholder, implemented in Part 2)
# ═════════════════════════════════════════════════════════════════════════════

def _tab_training() -> None:
    st.markdown(
        _html_phase_header(
            "3", "Model Training",
            "Train a binary classifier on the QUBO-reduced feature set."
        ),
        unsafe_allow_html=True,
    )

    if not st.session_state.feature_selection_done:
        st.markdown(
            _html_lock_banner(
                "Complete <strong>Phase 2 — Feature Selection</strong> "
                "before training a classifier."
            ),
            unsafe_allow_html=True,
        )
        return

    # ── Will be fully implemented in Part 2 ───────────────────────────────
    st.markdown(
        '<div class="info-box">'
        "⚙️ &nbsp;Training controls will be available in <strong>Part 2</strong> of the GUI. "
        "The backend <code>train()</code> function is already implemented in "
        "<code>src/qubo_project/model.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    # Show what Phase 2 produced, so the user can verify before training
    if st.session_state.feature_selection_stats:
        with st.expander("📋 Feature selection summary (Phase 2 output)", expanded=True):
            fs = st.session_state.feature_selection_stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Selected features", fs.get("n_selected", "—"))
            col2.metric("Training samples",  f"{fs.get('training_dataset_size', 0):,}")
            col3.metric("Test samples",       f"{fs.get('test_dataset_size',     0):,}")
            st.caption(
                f"Algorithm: `{fs.get('algorithm', 'N/A')}` · "
                f"Seed: `{fs.get('seed', 'N/A')}` · "
                f"Best α: `{fs.get('alpha', 0):.4f}`"
            )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — PREDICTION  (placeholder, implemented in Part 2)
# ═════════════════════════════════════════════════════════════════════════════

def _tab_prediction() -> None:
    st.markdown(
        _html_phase_header(
            "4", "Prediction & Evaluation",
            "Apply the trained classifier to the test set and inspect performance metrics."
        ),
        unsafe_allow_html=True,
    )

    if not st.session_state.training_done:
        st.markdown(
            _html_lock_banner(
                "Complete <strong>Phase 3 — Training</strong> before running predictions."
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        '<div class="info-box">'
        "⚙️ &nbsp;Prediction controls will be available in <strong>Part 2</strong> of the GUI. "
        "The backend <code>predict()</code> function is already implemented in "
        "<code>src/qubo_project/model.py</code>."
        "</div>",
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    _render_sidebar()

    # ── Main header ───────────────────────────────────────────────────────
    st.markdown(
        '<div style="margin-bottom:1.4rem;">'
        '  <h1 class="page-title">'
        '    QUBO Feature Selection &amp; Classification Pipeline'
        '  </h1>'
        '  <p class="page-subtitle">'
        '    Binary credit-risk classification via QUBO-optimised feature reduction '
        '    and machine learning'
        '  </p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️  Phase 1 · Preprocessing",
        "🔬  Phase 2 · Feature Selection",
        "🤖  Phase 3 · Training",
        "📊  Phase 4 · Prediction",
    ])

    with tab1:
        _tab_preprocessing()
    with tab2:
        _tab_feature_selection()
    with tab3:
        _tab_training()
    with tab4:
        _tab_prediction()


if __name__ == "__main__":
    main()