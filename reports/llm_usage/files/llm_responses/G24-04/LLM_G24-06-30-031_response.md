Now I have a complete picture of the full codebase. Let me write the first half of the GUI:The file is clean — 1,122 lines, zero syntax errors, 21 functions defined. Here's a full walkthrough of every architectural decision:

---

## Design decisions

**Visual identity** — I chose a deep-charcoal / electric-teal palette (`#0D1117` background, `#00C9A7` accent) over the generic cream-serif or acid-green defaults. This reads as "precision data tooling" rather than "generic ML demo", which fits the academic evaluation context.

**Signature element** — The **pipeline status tracker** in the sidebar. It renders four numbered step-badges that animate through `pending → active → done` states using only CSS class swaps, giving the professor an at-a-glance view of where a run stands. This is genuinely functional, not decorative.

**Phase gating with `session_state`** — Every tab checks its predecessor's completion flag before rendering anything. The `_html_lock_banner()` fragment shows a warm amber warning explaining *exactly* what to do first, rather than a cold disabled state. Uploading a new dataset automatically invalidates all downstream phases.

**Clean function architecture** — Each phase is split into three responsibilities: `_tab_X()` (layout + parameters), `_run_X()` (backend call + error handling + state write), and `_render_X_results()` (metrics display). This makes Part 2 (Training + Prediction) a simple copy-paste extension with no refactoring.

**Alpha search chart** — The Plotly chart in Phase 2 shows the bisection trajectory with the acceptable K±allowance band shaded in blue and the winning α marked with a dashed green line. It falls back gracefully to `st.dataframe` if plotly is missing.

**Error handling** — Three distinct failure modes are caught: `FileNotFoundError`, `ValueError` (e.g. wrong target column), and a catch-all `Exception`. Each produces a specific, actionable error message. The sidebar also shows a banner if backend imports fail entirely.

**Target column is global** — Stored in `st.session_state.target_column` and set once in the sidebar, so it's consistent across all four phases without the user re-typing it.