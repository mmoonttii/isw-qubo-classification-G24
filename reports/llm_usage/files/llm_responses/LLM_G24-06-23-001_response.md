Here is a comprehensive roadmap for your Software Engineering project, followed by the first prompt you should send to Claude to kickstart the development.

Since your assignment strictly requires LLM interactions to be in English, the roadmap and the prompt are written entirely in English. You can also use this roadmap as a reference to keep track of your progress and the markdown logs you need to submit.

### Project Roadmap

**Phase 1: Project Setup & Scaffolding (Days 1-2)**

*   **Create the GitHub Repository**: Name it `isw-qubo-classification-GXX` (replace GXX with your group number).
*   **Build the Directory Structure**: Create the exact folder hierarchy requested (`data/`, `src/qubo_project/`, `tests/`, `outputs/`, `reports/`, etc.).
*   **Initialize Configuration Files**: Create `group_info.yaml`, `README.md`, and start tracking LLM interactions in `reports/llm_usage/LOG-GXX-01.md`.
*   **Virtual Environment**: Setup a Python 3.11+ virtual environment and initialize a `requirements.txt`.

**Phase 2: Preprocessing Module (Days 3-4)**
*   **Implement `preprocessing.py`**: 
    *   Write the `fit_normalize` function.
    *   Implement logic to drop columns with a high percentage of zeros/nulls (based on `minPercValid`).
    *   Implement Z-score normalization (ignoring the target column).
    *   Generate the specific JSON output schema.
*   **CLI Implementation**: Add `argparse` to support the required terminal commands.
*   **Testing**: Write `pytest` scripts in the `tests/` folder to verify preprocessing.

**Phase 3: QUBO Feature Selection (Days 5-7)**
*   **Mathematical Modeling**: Write the logic to compute the Spearman correlation matrix between features, and between features and the target.
*   **QUBO Matrix (Q) Construction**: Build the $Q$ matrix combining influence (target correlation weighted by $\alpha$) and independence (inter-feature correlation weighted by $1-\alpha$).
*   **Optimization Loop**: Implement `select_features` in `feature_selection.py`. Integrate an open-source QUBO solver (e.g., `dimod` with Simulated Annealing). Create a loop that varies $\alpha$ to reach the requested `percSelected` features within the `allowance`.
*   **CLI & Output**: Ensure the dataset is split (train/test), save the CSVs, and generate the strictly formatted JSON statistics.

**Phase 4: Model Training and Prediction (Days 8-10)**
*   **Implement `model.py` (Training)**:
    *   Implement the `train` function supporting at least 3 models (Random Forest is mandatory; others could be Logistic Regression, SVM, etc.).
    *   Save the model using `joblib`.
    *   Output the required JSON metrics (training time, dataset size, etc.).
*   **Implement `model.py` (Prediction)**:
    *   Implement the `predict` function.
    *   Generate the CSV with `row_n, target, prediction, score`.
    *   Compute and export the detailed JSON metrics (Precision, Recall, F1, ROC-AUC, Confusion Matrix).
*   **CLI Integration**: Add subcommands (`train` and `predict`) to `model.py` via `argparse`.

**Phase 5: Automated Testing (Day 11)**
*   **Pytest Suite**: Complete the `tests/` folder. Ensure tests verify:
    1. Preprocessing outputs numeric-only data and handles NaNs.
    2. Normalization validity.
    3. Feature selection outputs a binary vector and matches the ~20% requirement.
    4. Model saves properly and predictions yield a valid CSV.

**Phase 6: Graphical User Interface (Day 12)**
*   **Build the GUI**: Use `Streamlit` (highly recommended for ease of use) or `Gradio` to create `gui.py`. 
*   **Features**: Allow the user to upload/select a dataset, run preprocessing, feature selection, training, and prediction, while displaying outputs on the screen.

**Phase 7: Final Review & Documentation (Days 13-14)**
*   **Code Cleanup**: Ensure no absolute paths exist, `seed` is used everywhere for reproducibility, and `requirements.txt` is up-to-date.
*   **Documentation**: Fill out `project_report.yaml`. Ensure all `LOG-GXX-NN.md` files are properly formatted with your Claude prompts and responses.
*   **Delivery**: Verify the `.zip` download from GitHub works flawlessly.

---

### First Prompt for Claude Sonnet

*Copy and paste the following text exactly as it is to start your first session with Claude. Remember to save this interaction in your first Markdown log file!*

**Prompt:**

```text
Act as an expert Software Engineer and Python Data Scientist. I am working on a university project about "Binary Classification with QUBO Feature Reduction" using Python 3.11+. We must communicate STRICTLY in English. 

Here is the context and the strict constraints for the project:
1. We must create a specific folder structure:
isw-qubo-classification-GXX/
├── README.md
├── requirements.txt
├── data/
├── src/
│   └── qubo_project/
│       ├── __init__.py
│       ├── preprocessing.py
│       ├── feature_selection.py
│       ├── model.py
│       └── gui.py
├── tests/
└── outputs/

2. I need your help to write the first module: `src/qubo_project/preprocessing.py`. 
It must contain a specific function signature:
```python
def fit_normalize(
    input_csv: str, 
    target_column: str, 
    normalized_csv: str, 
    outInitalRes_json: str, 
    minPercValid: float = 0.05
):
```
This function must:
- Read a CSV dataset (handling missing values).
- Separate the target column (which contains binary 0/1 values).
- Drop features (columns) that have less than `minPercValid` percentage of valid, non-zero data (i.e., drop columns that are mostly zeros or NaNs).
- Apply Z-score normalization to the remaining features.
- Save the resulting normalized dataset to `normalized_csv` keeping the original headers and the target column.
- Save a JSON file to `outInitalRes_json` containing exactly these keys: "n_input_features", "n_kept_features", "dataset_size", "dataset_input_time", "dataset_processing_time", and "dropped_feature_names".

3. The file `preprocessing.py` must also be executable from the command line using `argparse` with this exact syntax:
```bash
python preprocessing.py \
  --input dati_credito.csv \
  --target target \
  --out-data normalized.csv \
  --out-json preprocessing_result.json \
  --min-perc-valid 0.06
```

Please provide:
1. An initial list of libraries I should put in `requirements.txt` for this data processing step (e.g., pandas, scikit-learn).
2. The complete, production-ready code for `src/qubo_project/preprocessing.py` fulfilling all the requirements, including docstrings and the CLI implementation. Ensure reproducibility and no absolute paths.
```