You are in the final stretch of your project! Building the GUI is a great way to tie all those complex backend modules together into something tangible and impressive.

Since you are aiming for that bonus point for the "best" GUI, **I strongly recommend choosing Streamlit**. While libraries like Tkinter are functional, Streamlit is specifically designed for machine learning and data science web apps. It looks incredibly professional right out of the box, makes displaying dataframes and JSON metrics very easy, and handles interactive elements beautifully. Plus, the specifications explicitly mention it as an option and note that it should be launchable via `streamlit run src/qubo_project/gui.py`.

Because you are still using the free tier of Claude Sonnet, asking it to write a complete, robust, error-handling Streamlit dashboard with four different phases in a single prompt will almost certainly result in truncated code or skipped requirements.

Here is a two-step prompting strategy to ensure Claude generates a robust, high-quality GUI that handles errors gracefully and covers all the required operations.

---

### Step 1: Core Layout, Preprocessing, and Feature Selection

This first prompt establishes the Streamlit architecture, sets up a sidebar for dataset selection, and implements the first two phases of your pipeline. Copy and paste this into Claude:

> Act as an expert Python UI Developer. I am building the final component of my university project: a Streamlit GUI for a QUBO feature reduction and classification pipeline. My professor is awarding a bonus point for the "best" GUI, so it needs to look highly professional, be user-friendly, and include robust error handling for bad inputs or missing data.
> Please write the first half of the `src/qubo_project/gui.py` file. Because of output limits, we will do this in two steps.
> **Requirements for Step 1:**
> 1. Use Streamlit to create a professional layout. Create a sidebar for global actions (like selecting/uploading the dataset) and use `st.tabs` in the main view for the 4 phases: Preprocessing, Feature Selection, Training, Prediction.
> 2. Implement the **Dataset Selection** in the sidebar. The user must be able to load a CSV. Include basic error handling (e.g., using `st.error` or `st.warning` if the file is invalid or not loaded).
> 3. Implement the **Preprocessing Tab**. Include input fields for `target_column` (text) and `minPercValid` (float). Add a button to "Start Preprocessing". When clicked, it should call the backend `fit_normalize` function (assume it is importable from `src.qubo_project.preprocessing`). Display a success message and show the resulting `outInitalRes_json` metrics nicely using Streamlit's JSON or metric components.
> 4. Implement the **Feature Selection Tab**. Include input fields for `percSelected` (float), `allowance` (int), `seed` (int), and `alpha_computations` (int). Add a "Start Feature Selection" button that calls `select_features` (from `src.qubo_project.feature_selection`). Provide a progress spinner or bar during execution, and display the output statistics visually once complete.
> 5. Ensure that the UI prevents the user from starting Phase 2 if Phase 1 hasn't been completed successfully (manage this using `st.session_state`).
> 
> 

---

### Step 2: Training, Prediction, and Visualizations

Once Claude provides a working script for the first half, feed it this second prompt to complete the application.

> This looks fantastic and very professional. Now, let's finish `src/qubo_project/gui.py` by implementing the last two tabs for Training and Prediction. Please provide the code to append to or update our existing Streamlit script.
> **Requirements for Step 2:**
> 1. Implement the **Training Tab**. Include a dropdown menu to let the user select the classifier (e.g., "random_forest", "logistic_regression"). Add a "Start Training" button that calls the `train` function (from `src.qubo_project.model`). Once finished, beautifully display the `training_metrics.json` results, highlighting the training time and dataset size.
> 2. Implement the **Prediction Tab**. Add a "Run Prediction" button that calls the `predict` function (from `src.qubo_project.model`).
> 3. **Visualizations:** The prediction output (`classif_stats_json`) contains metrics like accuracy, ROC-AUC, and a confusion matrix. Use Streamlit's native charting capabilities (like `st.bar_chart`, `st.dataframe`, or `st.metric`) to display these classification statistics in a visually appealing way.
> 4. Add a feature to let the user display or download the final `predictions.csv` file directly from the GUI.
> 5. Continue using `st.session_state` to strictly enforce the pipeline order (e.g., show a warning if the user tries to click "Start Training" before "Feature Selection" is completed). Keep the error handling robust.
> 
>