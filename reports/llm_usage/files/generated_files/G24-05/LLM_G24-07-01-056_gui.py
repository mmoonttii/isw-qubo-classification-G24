"""
gui.py — Streamlit front-end for the QUBO Feature Selection & Classification Pipeline.

Launch with:
    streamlit run src/qubo_project/gui.py
or:
    python -m qubo_project.gui

─────────────────────────────────────────────────────────────────────────────
Implements:
  • Sidebar  : dataset upload, pipeline status tracker, global controls
  • Phase 1  : Preprocessing       (fit_normalize)
  • Phase 2  : Feature Selection   (select_features + plotly alpha chart)
  • Phase 3  : Training            (train)
  • Phase 4  : Prediction          (predict + classification report charts)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
from streamlit_theme import st_theme
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
# ═════════════════════════════════════════════════════════════════════════════
# THEME DETECTION
# ═════════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: this app used to theme itself with @media (prefers-color-scheme),
# which only reacts to the OS/browser preference. Streamlit has its OWN theme
# switcher (☰ menu → Settings → Theme), independent of the OS setting, and
# hosting environments often pin one explicitly. When the in-app theme and the
# OS preference disagree, the media query fires for the wrong one — Streamlit
# repaints the page background itself (light), while our custom CSS stayed on
# its dark-mode default (light-on-light text). We ask Streamlit directly
# instead, so the two can never disagree.

def _detect_theme_base() -> str:
    """
    Detect the active Streamlit theme ('light' or 'dark').
    Uses `st_theme()` to force a rerun if the user changes the theme in the UI.
    """
    # st_theme() returns None on the very first microsecond of load, 
    # then returns a dict containing the active theme config.
    theme_info = st_theme()
    
    if theme_info is not None and "base" in theme_info:
        return theme_info["base"]
    
    # Fallback for the initial load before JS responds
    try:
        if st.context.theme.type in ("light", "dark"):
            return st.context.theme.type
    except Exception:
        pass
        
    return "dark"


# ═════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — injected as a single <style> block
# Theme:  deep-charcoal / electric-teal (dark)  ·  off-white / forest-teal (light)
# Signature element: pipeline status tracker with animated step badges
# ═════════════════════════════════════════════════════════════════════════════

# All colour decisions live here, keyed by the value _detect_theme_base()
# returns. A matching :root block is generated at render time — deterministic,
# and always in sync with whatever Streamlit is actually displaying.
_THEME_TOKENS: dict = {
    "dark": {
        # Accent — electric teal
        "accent":      "#00C9A7",
        "accent-glow": "rgba(0,201,167,.16)",
        "accent-edge": "rgba(0,201,167,.30)",
        # Surfaces — semi-transparent so they float above any bg
        "card-bg":     "rgba(255,255,255,.045)",
        "card-border": "rgba(255,255,255,.09)",
        # Typography
        "text-primary": "#E6EDF3",
        "text-muted":   "#8B949E",
        # Tab bar
        "tab-bar-bg": "rgba(255,255,255,.04)",
        "tab-color":  "rgba(255,255,255,.48)",
        # Status — ok / warning / info / error
        "ok-clr": "#3FB950",   "ok-bg":   "rgba(63,185,80,.11)",
        "warn-clr": "#F0883E", "warn-bg": "rgba(240,136,62,.09)", "warn-bd": "rgba(240,136,62,.28)",
        "blue-clr": "#58A6FF", "info-bg": "rgba(88,166,255,.08)", "info-bd": "rgba(88,166,255,.22)",
        "err-clr": "#FF6B6B",
        # Pipeline step states
        "ps-done-bg":       "rgba(63,185,80,.10)",
        "ps-pending-clr":   "rgba(255,255,255,.38)",
        "ps-pending-badge": "rgba(255,255,255,.10)",
        # Text placed ON TOP of var(--accent) — bright teal → black reads best.
        "accent-contrast": "#000",
    },
    "light": {
        # Accent — teal shifted darker for contrast on white
        "accent":      "#0A8A74",
        "accent-glow": "rgba(10,138,116,.12)",
        "accent-edge": "rgba(10,138,116,.25)",
        # Surfaces — very subtle dark tint on white
        "card-bg":     "rgba(0,0,0,.03)",
        "card-border": "rgba(0,0,0,.10)",
        # Typography
        "text-primary": "#1F2328",
        "text-muted":   "#656D76",
        # Tab bar
        "tab-bar-bg": "rgba(0,0,0,.04)",
        "tab-color":  "rgba(0,0,0,.45)",
        # Status — ok / warning / info / error
        "ok-clr": "#1A7F37",   "ok-bg":   "rgba(26,127,55,.08)",
        "warn-clr": "#9A6700", "warn-bg": "rgba(154,103,0,.07)", "warn-bd": "rgba(154,103,0,.25)",
        "blue-clr": "#0969DA", "info-bg": "rgba(9,105,218,.07)", "info-bd": "rgba(9,105,218,.20)",
        "err-clr": "#CF222E",
        # Pipeline step states
        "ps-done-bg":       "rgba(26,127,55,.08)",
        "ps-pending-clr":   "rgba(0,0,0,.35)",
        "ps-pending-badge": "rgba(0,0,0,.08)",
        # Light-mode accent (#0A8A74) is a darkened teal, sized for
        # text-on-white contrast — it is NOT light, so it needs a
        # light label on top of it, not a dark one.
        "accent-contrast": "#fff",
    },
}


def _build_root_css(theme: str) -> str:
    """Render the :root{...} custom-property block for the given theme."""
    tokens = _THEME_TOKENS.get(theme, _THEME_TOKENS["dark"])
    declarations = "\n    ".join(f"--{name}: {value};" for name, value in tokens.items())
    return (
        "<style>\n:root {\n    "
        + declarations
        + "\n    --radius: 10px;"
          '\n    --mono:   "JetBrains Mono","Fira Code",ui-monospace,monospace;'
          "\n}\n"
    )


_CSS = """

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

    # Phase 3 — Training
    "training_done":  False,
    "training_stats": None,
    "model_path":     None,

    # Phase 4 — Prediction
    "prediction_done":  False,
    "prediction_stats": None,
    "predictions_csv":  None,   # str: path to outputs/predictions.csv

    # Internal bookkeeping — NOT a pipeline phase. Tracks the (name, size)
    # signature of the last UploadedFile we actually processed, so we can
    # tell "still the same upload, just another rerun" apart from "the user
    # picked a new/different file". See _render_sidebar().
    "uploaded_signature": None,
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
            # `st.file_uploader` returns the SAME UploadedFile object on
            # EVERY rerun for as long as it stays attached to the widget —
            # not only on the run where the user picked it. If we called
            # _handle_upload() unconditionally here, it would fire again on
            # every single rerun the app performs afterwards — including
            # the reruns _run_preprocessing()/_run_feature_selection() now
            # trigger on success — and _handle_upload() resets every
            # downstream phase flag to False. That is precisely what caused
            # Phase 2 to "complete in the backend, then reset in the UI".
            #
            # Fix: only treat it as a genuinely NEW upload if its (name,
            # size) signature differs from the last one we processed.
            signature = (uploaded.name, uploaded.size)
            if signature != st.session_state.uploaded_signature:
                _handle_upload(uploaded)
                st.session_state.uploaded_signature = signature

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
            # fit_normalize() now honors output paths literally (backend
            # no longer reroutes bare filenames into outputs/ itself), so
            # the GUI resolves them here, same as the CLI does.
            stats = fit_normalize(
                input_csv         = st.session_state.dataset_path,
                target_column     = target_column,
                normalized_csv    = str(_outputs_dir() / "normalized.csv"),
                outInitalRes_json = str(_outputs_dir() / "preprocessing_result.json"),
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
            return
        except ValueError as exc:
            st.error(f"**Validation error:** {exc}")
            return
        except Exception as exc:
            st.error(f"**Unexpected error:** {exc}")
            return

    # Success: force an immediate rerun so every widget that reads
    # `preprocessing_done` — most importantly the sidebar pipeline
    # tracker, which was already drawn before this function ran — picks
    # up the new state right away instead of waiting for the next
    # user-triggered interaction.
    st.rerun()


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
            # select_features() now honors output paths literally (backend
            # no longer reroutes bare filenames into outputs/ itself), so
            # the GUI resolves them here, same as the CLI does.
            select_features(
                normalized_csv    = st.session_state.normalized_csv,
                reducedTrain_csv  = str(_outputs_dir() / "training_reduced.csv"),
                reducedTest_csv   = str(_outputs_dir() / "test_reduced.csv"),
                output_ottim_csv  = str(_outputs_dir() / "optimizations.csv"),
                output_json       = str(_outputs_dir() / "feature_selection_result.json"),
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
        return
    except ValueError as exc:
        progress.empty()
        st.error(f"**Validation error:** {exc}")
        return
    except Exception as exc:
        progress.empty()
        st.error(f"**Unexpected error during QUBO optimisation:** {exc}")
        return

    # Success: same reasoning as _run_preprocessing() — sync the sidebar
    # (and everything else reading session_state) immediately, rather
    # than leaving Phase 3 / Phase 4 gates and the pipeline tracker
    # showing stale info until another widget interaction forces a redraw.
    st.rerun()


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


# NOTE: _detect_theme_base() is defined once, near the top of the file,
# and reused both for CSS injection (main()) and for this chart's palette.


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
# PHASE 3 — TRAINING
# ═════════════════════════════════════════════════════════════════════════════

# Human-readable labels for the classifier keys returned by the backend's
# CLASSIFIERS constant. Falls back to a title-cased version of the raw key
# for anything not listed here, so new classifiers added to the backend
# never break the dropdown.
_CLASSIFIER_LABELS: dict = {
    "random_forest":       "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting":   "Gradient Boosting",
}


def _classifier_options() -> list[str]:
    """
    Resolve the list of classifier keys to offer in the dropdown.

    Reads from the backend's own `CLASSIFIERS` constant (imported from
    model.py) so the GUI can never drift out of sync with what model.py
    actually supports. Falls back to the three classifiers required by the
    project spec if the backend import failed or the constant's shape is
    unexpected, so the tab never crashes outright.
    """
    if _BACKEND_OK:
        try:
            if isinstance(CLASSIFIERS, dict):
                return list(CLASSIFIERS.keys())
            return list(CLASSIFIERS)
        except Exception:
            pass
    return ["random_forest", "logistic_regression", "gradient_boosting"]


def _tab_training() -> None:
    st.markdown(
        _html_phase_header(
            "3", "Model Training",
            "Train a binary classifier on the QUBO-reduced feature set."
        ),
        unsafe_allow_html=True,
    )

    # ── Phase gate ─────────────────────────────────────────────────────────
    if not st.session_state.feature_selection_done:
        st.markdown(
            _html_lock_banner(
                "Complete <strong>Phase 2 — Feature Selection</strong> "
                "before training a classifier."
            ),
            unsafe_allow_html=True,
        )
        return

    # Show what Phase 2 produced, so the user can verify before training
    if st.session_state.feature_selection_stats:
        with st.expander("📋 Feature selection summary (Phase 2 output)", expanded=False):
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

    # ── Parameter card ─────────────────────────────────────────────────────
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    st.markdown("<h4>⚙️ Training Parameters</h4>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        classifier_options = _classifier_options()
        classifier = st.selectbox(
            "Classifier",
            options=classifier_options,
            format_func=lambda key: _CLASSIFIER_LABELS.get(key, key.replace("_", " ").title()),
            help=(
                "One of the three binary classifiers implemented in model.py. "
                "Random Forest is the classifier required by the project spec."
            ),
        )

    with col_b:
        seed = int(st.number_input(
            "Random seed",
            min_value=0, max_value=2**31 - 1, value=42, step=1,
            help="Seed passed to the classifier constructor for reproducible training.",
        ))

    st.markdown(
        f'<div class="info-box">'
        f'Training set: &nbsp;<code>{st.session_state.reduced_train_csv or "—"}</code><br/>'
        f'Target column: &nbsp;<strong><code>{st.session_state.target_column}</code></strong>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close param-card

    # ── Run ────────────────────────────────────────────────────────────────
    if not _BACKEND_OK:
        st.error(f"Backend import error — cannot run: `{_BACKEND_ERR}`")
        return

    if st.button("▶  Start Training", type="primary", key="btn_train"):
        _run_training(classifier=classifier, seed=seed)

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.training_done and st.session_state.training_stats:
        _render_training_results(st.session_state.training_stats)


def _run_training(classifier: str, seed: int) -> None:
    """Call the backend train() function and load the resulting metrics."""
    label = _CLASSIFIER_LABELS.get(classifier, classifier.replace("_", " ").title())

    with st.spinner(f"Training the {label} classifier …"):
        try:
            # train() now honors output paths literally (backend no
            # longer reroutes bare filenames into outputs/ itself), so
            # the GUI resolves them here, same as the CLI does.
            train(
                classifier       = classifier,
                reducedTrain_csv = st.session_state.reduced_train_csv,
                target_column    = st.session_state.target_column,
                model_path       = str(_outputs_dir() / "model.joblib"),
                metrics_json     = str(_outputs_dir() / "training_metrics.json"),
                seed             = seed,
            )

            # Read the metrics produced by the backend, as the spec requires,
            # rather than relying on an in-memory return value.
            json_path = _outputs_dir() / "training_metrics.json"
            with open(json_path, encoding="utf-8") as fh:
                metrics = json.load(fh)

            st.session_state.training_done  = True
            st.session_state.training_stats = metrics
            st.session_state.model_path     = str(_outputs_dir() / "model.joblib")

            # Invalidate the downstream phase — a fresh model means any
            # previous prediction results are now stale.
            st.session_state.prediction_done  = False
            st.session_state.prediction_stats = None

        except FileNotFoundError as exc:
            st.error(f"**File not found:** {exc}")
            return
        except ValueError as exc:
            st.error(f"**Validation error:** {exc}")
            return
        except Exception as exc:
            st.error(f"**Unexpected error during training:** {exc}")
            return

    # Success: same reasoning as the earlier phases — sync the sidebar
    # pipeline tracker and the Phase 4 gate immediately, rather than
    # waiting for the next user-triggered rerun.
    st.rerun()


def _render_training_results(stats: dict) -> None:
    model_name = Path(stats.get("model_path", "model.joblib")).name
    st.success(
        f"✓ Training complete — **{stats.get('classifier', '?')}** model saved to "
        f"`outputs/{model_name}`."
    )

    # ── Headline metrics, via native st.metric / columns ────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Time", f"{stats.get('training_time', 0):.2f} s")
    col2.metric("Training Samples", f"{stats.get('n_samples', 0):,}")
    col3.metric("Features Used", stats.get("n_features", "—"))

    # ── Secondary metric grid, consistent with the other phases ────────────
    grid_html = _html_metric_grid(
        _html_metric_card("Classifier",     stats.get("classifier", "—")),
        _html_metric_card("Seed",           stats.get("seed", "—")),
        _html_metric_card("Target = 1",     f"{stats.get('target_1_percentage', 0):.2f}", "%"),
        _html_metric_card("Read Time",      f"{stats.get('dataset_input_time', 0):.2f}", "s"),
    )
    st.markdown(grid_html, unsafe_allow_html=True)

    st.caption(
        f"Model file: &nbsp;`{stats.get('model_path', '—')}` · "
        f"Training dataset: &nbsp;`{stats.get('training_dataset', '—')}` · "
        f"Target column: &nbsp;`{stats.get('target_column', '—')}`"
    )

    with st.expander("📄 Full training metrics (JSON)", expanded=False):
        st.json(stats)


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 4 — PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def _tab_prediction() -> None:
    st.markdown(
        _html_phase_header(
            "4", "Prediction & Evaluation",
            "Apply the trained classifier to the test set and inspect performance metrics."
        ),
        unsafe_allow_html=True,
    )

    # ── Phase gate ─────────────────────────────────────────────────────────
    if not st.session_state.training_done:
        st.markdown(
            _html_lock_banner(
                "Complete <strong>Phase 3 — Training</strong> before running predictions."
            ),
            unsafe_allow_html=True,
        )
        return

    # Show what Phase 3 produced, so the user can verify before predicting
    if st.session_state.training_stats:
        with st.expander("📋 Training summary (Phase 3 output)", expanded=False):
            ts = st.session_state.training_stats
            col1, col2, col3 = st.columns(3)
            col1.metric("Classifier",   ts.get("classifier", "—"))
            col2.metric("Train Samples", f"{ts.get('n_samples', 0):,}")
            col3.metric("Features Used", ts.get("n_features", "—"))
            st.caption(f"Model file: `{st.session_state.model_path or '—'}`")

    # ── Inputs card (read-only — everything comes from earlier phases) ─────
    st.markdown('<div class="param-card">', unsafe_allow_html=True)
    st.markdown("<h4>⚙️ Prediction Inputs</h4>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="info-box">'
        f'Test set: &nbsp;<code>{st.session_state.reduced_test_csv or "—"}</code><br/>'
        f'Model: &nbsp;<code>{st.session_state.model_path or "—"}</code><br/>'
        f'Target column: &nbsp;<strong><code>{st.session_state.target_column}</code></strong>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # close param-card

    # ── Inline validation (defensive — phase gate above should prevent this,
    # but session state can be edited externally, so we double-check) ──────
    warnings_list: list[str] = []
    if not st.session_state.reduced_test_csv:
        warnings_list.append(
            "No reduced test set found — re-run **Phase 2 — Feature Selection**."
        )
    if not st.session_state.model_path:
        warnings_list.append(
            "No trained model found — re-run **Phase 3 — Training**."
        )
    for w in warnings_list:
        st.warning(w)

    # ── Run ────────────────────────────────────────────────────────────────
    if not _BACKEND_OK:
        st.error(f"Backend import error — cannot run: `{_BACKEND_ERR}`")
        return

    run_disabled = bool(warnings_list)
    if st.button(
        "▶  Run Prediction",
        type="primary",
        disabled=run_disabled,
        key="btn_predict",
    ):
        _run_prediction()

    # ── Results ────────────────────────────────────────────────────────────
    if st.session_state.prediction_done and st.session_state.prediction_stats:
        _render_prediction_results(st.session_state.prediction_stats)


def _run_prediction() -> None:
    """Call the backend predict() function and load the resulting statistics."""
    with st.spinner("Classifying the test set …"):
        try:
            # predict() now honors output paths literally (backend no
            # longer reroutes bare filenames into outputs/ itself), so
            # the GUI resolves them here, same as the CLI does.
            predict(
                reduced_Test_csv  = st.session_state.reduced_test_csv,
                target_column     = st.session_state.target_column,
                model_path        = st.session_state.model_path,
                predictions_csv   = str(_outputs_dir() / "predictions.csv"),
                classif_stats_json= str(_outputs_dir() / "classification_stats.json"),
            )

            # Read the statistics produced by the backend, as in the other
            # phases, rather than relying on an in-memory return value.
            json_path = _outputs_dir() / "classification_stats.json"
            with open(json_path, encoding="utf-8") as fh:
                stats = json.load(fh)

            st.session_state.prediction_done  = True
            st.session_state.prediction_stats = stats
            st.session_state.predictions_csv  = str(_outputs_dir() / "predictions.csv")

        except FileNotFoundError as exc:
            st.error(f"**File not found:** {exc}")
            return
        except ValueError as exc:
            st.error(f"**Validation error:** {exc}")
            return
        except Exception as exc:
            st.error(f"**Unexpected error during prediction:** {exc}")
            return

    # Success: same reasoning as the earlier phases — sync the sidebar
    # pipeline tracker immediately, rather than waiting for the next
    # user-triggered rerun.
    st.rerun()


def _render_prediction_results(stats: dict) -> None:
    st.success(
        f"✓ Prediction complete — **{stats.get('n_samples', 0):,}** test samples classified "
        f"with **{stats.get('classifier', '?')}**."
    )

    # ── Headline metrics ─────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{stats.get('accuracy', 0):.4f}")
    col2.metric("ROC-AUC",  f"{stats.get('roc_auc', 0):.4f}")
    col3.metric("Total Samples", f"{stats.get('n_samples', 0):,}")
    col4.metric("Target = 1 (%)", f"{stats.get('target_1_percentage', 0):.2f}")

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

    chart_col, matrix_col = st.columns([1.15, 1], gap="large")

    # ── Per-class precision / recall / F1 — grouped bar chart ──────────────
    with chart_col:
        st.markdown("**Classification Report**")
        class_0 = stats.get("class_0", {})
        class_1 = stats.get("class_1", {})
        report_df = pd.DataFrame(
            {
                "Class 0": [
                    class_0.get("precision", 0),
                    class_0.get("recall", 0),
                    class_0.get("f1", 0),
                ],
                "Class 1": [
                    class_1.get("precision", 0),
                    class_1.get("recall", 0),
                    class_1.get("f1", 0),
                ],
            },
            index=["Precision", "Recall", "F1"],
        )
        st.bar_chart(report_df, height=300)
        st.caption(
            f"Support — Class 0: **{class_0.get('support', '—')}** · "
            f"Class 1: **{class_1.get('support', '—')}**"
        )

    # ── Confusion matrix — styled dataframe ─────────────────────────────────
    with matrix_col:
        st.markdown("**Confusion Matrix**")
        cm = stats.get("confusion_matrix", {})
        labels = cm.get("labels", [0, 1])
        matrix = cm.get("matrix", [[0, 0], [0, 0]])
        cm_df = pd.DataFrame(
            matrix,
            index=[f"Actual {l}" for l in labels],
            columns=[f"Predicted {l}" for l in labels],
        )
        try:
            styled = cm_df.style.background_gradient(cmap="Blues").format("{:,}")
            st.dataframe(styled, use_container_width=True)
        except ImportError:
            # matplotlib not installed — fall back to a plain table
            st.dataframe(cm_df, use_container_width=True)
        st.caption("Rows = actual class · Columns = predicted class.")

    # ── Full JSON ─────────────────────────────────────────────────────────
    with st.expander("📄 Full classification statistics (JSON)", expanded=False):
        st.json(stats)

    # ── Predictions file: preview + download ────────────────────────────────
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    st.markdown("**Predictions Output**")

    pred_path_str = st.session_state.predictions_csv
    if not pred_path_str:
        st.info("No predictions file available yet.")
        return

    pred_path = Path(pred_path_str)
    if not pred_path.exists():
        st.warning(f"Predictions file not found at `{pred_path}`.")
        return

    file_size_kb = pred_path.stat().st_size / 1024
    st.caption(f"`{pred_path.name}` · {file_size_kb:,.1f} KB")

    only_misclassified = st.checkbox(
        "Show only misclassified rows (target ≠ prediction)",
        value=False,
        help="Filters the preview below to rows where the model got it wrong.",
    )

    try:
        preview_df = pd.read_csv(pred_path, nrows=500)
        if only_misclassified and {"target", "prediction"}.issubset(preview_df.columns):
            preview_df = preview_df[preview_df["target"] != preview_df["prediction"]]
        st.dataframe(preview_df, use_container_width=True, height=260)
        st.caption(
            "Showing up to the first 500 rows. Download the full file below for "
            "the complete result set."
        )
    except Exception as exc:
        st.warning(f"Could not preview predictions file: {exc}")

    # Stream the raw bytes from disk for the download — avoids loading the
    # full (potentially multi-million-row) CSV into memory just to re-encode it.
    try:
        st.download_button(
            "⬇  Download predictions.csv",
            data=pred_path.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not prepare the download: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()

    # Detect Streamlit's actual rendered theme once per run, and use it
    # for both the injected CSS and any Python-side rendering (e.g. the
    # Plotly chart) — this is the single source of truth for "light vs
    # dark", replacing the old @media (prefers-color-scheme) guesswork.
    theme = _detect_theme_base()
    st.markdown(_build_root_css(theme) + _CSS, unsafe_allow_html=True)

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