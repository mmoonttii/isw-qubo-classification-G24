Hey there! It looks like you have hit the classic Streamlit `session_state` wall. Managing state across a multi-step pipeline can be incredibly frustrating, but you are definitely not alone in running into this.

Before we jump into the fix, just a quick factual heads-up: you mentioned "Claude Sonnet 4.6," but the current state-of-the-art from Anthropic is **Claude 3.5 Sonnet**. I will assume that is the model you are using and tailor the prompt accordingly.

I took a look at the `gui.py` file you provided, and I can tell you exactly why this is happening so you understand the underlying mechanics before Claude writes the code:

1. **The Total Reset (Phase 2 Bug):** Streamlit runs your script from top to bottom every time a button is clicked. When you click "Start Feature Selection," the script reruns. Because `st.file_uploader` still holds your uploaded file in memory, `if uploaded is not None:` evaluates to `True` again. This triggers `_handle_upload(uploaded)` on every single click, which ruthlessly overwrites your `session_state` and wipes out all your Phase 1 progress.
2. **The Sidebar Sync (Phase 1 Bug):** Streamlit renders elements sequentially. `_render_sidebar()` is called *before* the tabs and your "Start Preprocessing" button. When preprocessing finishes and updates the `session_state`, the sidebar has already been drawn for that run. It won't show the new status until the *next* time the app reruns.

Here is the exact prompt you should feed into Claude to get this fixed.

---

### Prompt for Claude 3.5 Sonnet

**Copy and paste the text below into your Claude chat:**

> Act as an expert Python UI Developer specializing in Streamlit. I need you to fix two critical state management bugs in my `gui.py` file for my QUBO Feature Selection pipeline.
> **The Problems:**
> 1. **Pipeline Status Sync:** After Phase 1 (Preprocessing) completes successfully, the pipeline status tracker in the sidebar does not update immediately. It only updates if I trigger another action on the page.
> 2. **State Wipe on Button Click:** After Phase 1, when I click "Start Feature Selection" in Phase 2, the app completely resets. All Phase 1 session state data is lost, the metrics disappear, and the Phase 2 UI locks up, even though the backend console shows the Phase 2 QUBO algorithm ran successfully.
> 
> 
> **The Root Causes to Fix:**
> * **For Problem 1:** Streamlit renders the sidebar before the phase functions update the `session_state`. You need to add `st.rerun()` at the very end of successful `_run_preprocessing()` and `_run_feature_selection()` executions to force the UI (specifically the sidebar) to sync with the new state.
> * **For Problem 2:** The `st.file_uploader` in `_render_sidebar()` returns the uploaded file object on every rerun. Currently, `if uploaded is not None:` blindly calls `_handle_upload(uploaded)` every time a button is clicked. `_handle_upload` resets all phase states to `False`. You need to update the logic so that `_handle_upload` is **only** triggered if the uploaded file is *different* from the one currently stored in `st.session_state.dataset_name`.
> 
> 
> Please provide the corrected code blocks for `_render_sidebar()`, `_run_preprocessing()`, and `_run_feature_selection()`. Keep the rest of the existing styling and architecture intact.