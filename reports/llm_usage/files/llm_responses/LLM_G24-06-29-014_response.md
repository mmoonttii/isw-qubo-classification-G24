Hey there! Good luck with your Software Engineering project. Building a QUBO feature reduction classification pipeline is a pretty cool, albeit challenging, assignment!

Just a quick factual heads-up: the current state-of-the-art version of Claude is from the 3.5 family (specifically Claude 3.5 Sonnet), so I'll assume that is the model you are interacting with.

You are absolutely right to consider splitting the workload. Since `model.py` requires implementing machine learning training, prediction, complex JSON metric generation, and a Command Line Interface (CLI), asking an LLM to do all of it in one shot often leads to skipped requirements or messy code.

I recommend a **two-step approach**. First, we will ask Claude to build the core functions (`train` and `predict`). Once that is solid, we will ask it to add the CLI wrapper.

Here are the prompts, written strictly in English as required by your project constraints.

### Step 1: The Core Logic (`train` and `predict`)

Copy and paste this prompt into Claude to get the base logic working.

**Prompt 1:**

> Act as an expert Python Software Engineer. I am working on a university project for binary classification using QUBO feature reduction. I need to write the `model.py` module, which will handle Phase 3 (Learning) and Phase 4 (Prediction). The preprocessing and feature selection modules are already completed.
> Please implement the core logic for `src/qubo_project/model.py`. Use standard libraries like `pandas`, `scikit-learn`, `joblib`, and `json`.
> **Requirements for the `train` function:**
> 1. Signature exactly as follows: `def train(classifier: str, reducedTrain_csv: str, target_column: str, model_path: str, metrics_json: str, seed: int = 42):`
> 2. Implement 3 binary classifiers. One MUST be Random Forest (`"random_forest"`). Please choose two others (e.g., `"logistic_regression"`, `"gradient_boosting"`) and map the `classifier` string argument to the correct model.
> 3. Read the normalized and reduced training dataset (`reducedTrain_csv`), split features and the `target_column`.
> 4. Train the selected classifier using the provided `seed`.
> 5. Save the trained model to `model_path` using `.joblib`.
> 6. Save the training metrics to `metrics_json` strictly following this JSON structure (calculate times appropriately):
> 
> 
> ```json
> {
>   "classifier": "random_forest",
>   "seed": 42,
>   "training_dataset": "training_reduced.csv",
>   "target_column": "target",
>   "model_path": "model80.joblib",
>   "n_samples": 14000,
>   "n_features": 19,
>   "target_1_percentage": 1.59,
>   "dataset_input_time": 0.16,
>   "training_time": 1.61
> }
> 
> ```
> 
> 
> **Requirements for the `predict` function:**
> 1. Signature exactly as follows: `def predict(reduced_Test_csv: str, target_column: str, model_path: str, predictions_csv: str, classif_stats_json: str):`
> 2. Load the test dataset and the trained classifier from `model_path`.
> 3. Generate predictions and probabilities (scores).
> 4. Save a CSV to `predictions_csv` with exactly these columns: `row_n,target,prediction,score` (where `score` is the probability of the positive class).
> 5. Calculate and save classification stats to `classif_stats_json` strictly following this JSON structure:
> 
> 
> ```json
> {
>   "classifier": "random_forest",
>   "n_samples": 6000,
>   "target_1_count": 375,
>   "target_1_percentage": 1.56,
>   "accuracy": 0.9914,
>   "class_0": {"precision": 0.99, "recall": 0.99, "f1": 0.99, "support": 23625},
>   "class_1": {"precision": 0.86, "recall": 0.53, "f1": 0.66, "support": 375},
>   "roc_auc": 0.966,
>   "confusion_matrix": {
>     "labels": [0, 1],
>     "matrix": [[23593, 32], [174, 201]]
>   }
> }
> 
> ```
> 
> 
> Do not include the CLI execution logic yet, just the functions and necessary imports.

---

### Step 2: The Command Line Interface (CLI)

Once Claude gives you a good script for the logic, reply with this second prompt to add the terminal execution capabilities.

**Prompt 2:**

> This looks great. Now, I need to make `model.py` executable from the command line using the `argparse` module, as the automated evaluation system will call it via the terminal.
> Please update `model.py` to include an `if __name__ == "__main__":` block that parses arguments and calls the appropriate function (`train` or `predict`).
> It must support these exact two subcommands and arguments:
> **For training:**
> `python model.py train --classifier random_forest --in-reduced training_reduced.csv --target target --out-model model.joblib --out-metrics training_metrics.json --seed 42`
> **For predicting:**
> `python model.py predict --input-testset test_reduced.csv --target target --model model.joblib --out-predictions predictions.csv --out-stats classification_stats.json`
> Make sure the argument names match exactly (e.g., `--in-reduced`, `--out-stats`).

---

Which two classifiers (besides the mandatory Random Forest) are you planning to implement, or would you like Claude to pick them for you?