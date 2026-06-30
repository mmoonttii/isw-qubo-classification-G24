This is a classic Streamlit frustration. You have correctly diagnosed the problem: your custom CSS is injected by the Python backend based on the current theme, but Streamlit does not trigger a backend rerun when a user changes the theme via the Settings menu.

When the user toggles the theme, Streamlit's React frontend instantly updates its native CSS variables. However, your Python script doesn't know this happened until the user clicks a button or interacts with a widget (which forces a rerun). Until that rerun happens, your injected <style>:root {...}</style> block is stuck holding the color tokens from the previous theme.

Here is how you can fix this.
The Solution: streamlit-theme

Because there is no native way in Streamlit to force a backend rerun from the frontend Settings menu, the community relies on a lightweight third-party component called streamlit-theme.

This component uses JavaScript to listen for changes to the Streamlit theme on the frontend. When it detects a change, it automatically sends a signal back to Python, forcing an immediate rerun so your CSS can update seamlessly.
Step 1: Install the package
Bash

pip install streamlit-theme

Step 2: Update your _detect_theme_base function

You can completely replace your existing _detect_theme_base() logic with this. When st_theme() detects a change, it will automatically rerun the app, fetch the new dictionary of theme values, and your main() function will inject the correct CSS block.
Python

from streamlit_theme import st_theme
import streamlit as st

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

Why your previous @media query approach failed

As you rightly noted in your code comments, using @media (prefers-color-scheme) only checks the operating system's theme, not the application's internal state.

If you want to completely avoid third-party packages, the only other architectural solution is to map your custom CSS entirely to Streamlit's native CSS variables (e.g., var(--primary-color), var(--background-color), var(--text-color)). Because these are managed by the React frontend, they update instantly without a Python rerun.

However, looking at your _THEME_TOKENS, you have a highly customized, multi-layered palette (glows, edge colors, specific badge statuses). Streamlit simply doesn't expose enough native CSS variables to map all of those gracefully. Therefore, using streamlit-theme to force the Python rerun is the most robust way to maintain your complex design system.