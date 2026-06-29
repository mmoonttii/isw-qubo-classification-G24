It sounds like your project is really coming together! Automated testing is exactly what you need at this stage to ensure everything is robust before layering a GUI on top.

Just a quick factual reality check: you mentioned "Claude Sonnet 4.6," but the current state-of-the-art model from Anthropic is Claude 3.5 Sonnet. I’ll assume that’s the one you’re using on the free tier.

### **Effort Level Recommendation**

Since you are using the free tier of Claude and want to aim for that bonus point awarded to the *best* test suite, **you should split this workload into at least two steps.** Free tier LLMs can struggle with context windows and output limits. If you ask for a comprehensive, high-quality test suite encompassing four different phases of a pipeline in one shot, the LLM will likely give you shallow, generic tests just to fit the output limit. To win that bonus point, you need deep, robust tests (quality over quantity).

Here is a two-step prompting strategy, written strictly in English as your professor requires.

---

### **Step 1: Setup, Preprocessing, and Feature Selection Tests**

This first prompt sets up the `pytest` environment, the dummy data, and tests the first half of your pipeline.

**Copy and paste this into Claude:**

> Act as an expert Python Software Engineer in charge of Quality Assurance. I need to write a high-quality `pytest` suite for a QUBO feature reduction and binary classification pipeline I am building for a university project.
> My professor is awarding a bonus point for the best, most robust test suite, emphasizing quality over quantity. Please help me write the `tests/test_pipeline.py` file. Because of output limits, we will do this in two steps.
> **For this first step, focus ONLY on the fixtures, preprocessing, and feature selection.**
> **Requirements:**
> 1. Use standard `pytest` fixtures to generate a small, dummy `sample_test_dataset.csv` in a temporary directory. The dummy dataset must have numerical columns, some missing values (NaN), and a binary target column containing both `0` and `1` (at least 10% representation of each class).
> 2. Write a test to verify that the `preprocessing.py` module outputs ONLY numerical columns and successfully handles/manages missing values.
> 3. Write a test to verify that the normalization step produces a mathematically valid dataset (e.g., proper z-score normalization).
> 4. Write a test to verify that `feature_selection.py` (which uses QUBO) outputs a valid binary vector.
> 5. Write a test to verify that the number of selected features is approximately 20% of the original features.
> 
> 
> Please write clean, well-commented Python code using `pytest`. Use `tmp_path` or `tempfile` for file generation so the tests are self-contained.

---

### **Step 2: Model Training and Prediction Tests**

Once Claude has given you a solid set of tests for the first half, feed it this second prompt to test the machine learning components.

**Copy and paste this into Claude:**

> This looks excellent and highly robust. Now, let's complete the test suite by adding the tests for the machine learning model training and prediction phases (`model.py`).
> Please append the following tests to our `pytest` suite, reusing the fixtures we established where appropriate:
> **Requirements:**
> 1. Write a test to verify that calling the `train` function successfully produces and saves a model file (e.g., a `.joblib` file) to the disk.
> 2. Write a test to verify that calling the `predict` function (using the saved model and a reduced test dataset) produces a predictions CSV file.
> 3. Verify that this predictions CSV file contains exactly the required columns: `row_n`, `target`, `prediction`, and `score`.
> 4. Ensure there are edge-case assertions (e.g., verifying that the `score` is a valid probability between 0 and 1, and that `prediction` only contains 0 or 1).
> 
> 
> Please provide the code to append to our existing test file, maintaining the high standard of quality, clear assertions, and robust temporary file handling.