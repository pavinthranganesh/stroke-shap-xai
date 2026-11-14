# Stroke Classification and SHAP Explainability

This repository contains the Python exercises and an extended explainable AI (XAI) experiment using the **Stroke Prediction Dataset**. The final notebook compares a Random Forest classifier explained with Tree SHAP against an RBF-kernel Support Vector Machine (SVM) explained with Kernel SHAP.

## Project objectives

- Explore the stroke dataset and document data quality and class imbalance.
- Train and evaluate a Random Forest model using an 80/20 stratified split.
- Explain individual predictions and overall feature influence using Tree SHAP.
- Present explanations for technical, clinical, and non-technical audiences.
- Train an alternative RBF-SVM model and compare its performance and Kernel SHAP explanations with the Random Forest.

> **Important:** This is an educational machine-learning exercise, not a clinically validated stroke prediction or diagnostic tool. SHAP explains model predictions; it does not establish causal effects.

## Key files

| File | Description |
| --- | --- |
| `Stroke_SHAP_FinalPavin_Cleaned_Fixed.ipynb` | Main assignment notebook, including EDA, model evaluation, SHAP analysis, and visualizations. |
| `utils_improved.py` | Reusable improved stroke-data loading, splitting, preprocessing, and oversampling utilities. |
| `utils.py` | Data-loading utility used by the original exercise scripts. |
| `00_data_exploration.py` | Original exploratory data analysis exercise. |
| `01_interpretable_models.py` | Interpretable-model exercise. |
| `02_lime.py` | LIME explanation exercise. |
| `03_shap.py` | SHAP explanation exercise. |
| `04_counterfactuals.py` | Counterfactual explanation exercise. |
| `05_lrp.py` | Layer-wise relevance propagation / image-model exercise. |
| `data/README.md` | Original dataset source and setup notes. |
| `*.png` and `*.csv` | Saved figures and feature-importance results from the experiments. |

## Dataset

Stroke Prediction Dataset (Kaggle):
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

Download the CSV and place it at:

```text
data/healthcare-dataset-stroke-data.csv
```

The dataset contains 5,110 records and a binary `stroke` target. The experiment uses an 80/20 stratified train/test split (random state `2021`). Missing BMI values are imputed using a median fitted on training data only. Categorical variables are encoded, and random oversampling is applied **only to the training set**.

The `05_lrp.py` exercise uses a separate Brain MRI dataset. Refer to `data/README.md` for its source and expected `data/brain_mri/` folders. The Brain MRI images are not required for the main stroke notebook.

Check dataset licences and permissions before redistributing downloaded data or original teaching materials.

## Setup

Use a compatible Python environment and install the dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The dependency list is intentionally **not version-pinned**. For exact reproducibility, use the Python and package versions recorded by the reproducibility cell in the final notebook and verify that the environment can execute all cells.

## Run the assignment

1. Open the **project root** (`xai-series-master`) in VS Code or JupyterLab.
2. Confirm the stroke CSV exists at `data/healthcare-dataset-stroke-data.csv`.
3. Launch JupyterLab with `jupyter lab`, or open the notebook directly in VS Code.
4. Open `Stroke_SHAP_FinalPavin_Cleaned_Fixed.ipynb`.
5. Select the project Python kernel and run the cells in order.

The notebook invokes `00_data_exploration.py` and `03_shap.py` with `%run`, so those scripts must remain accessible in the project root. It also saves some generated figures and CSV results directly into that root folder.

**Note:** The current notebook defines `ImprovedDataLoader` within a code cell. A matching implementation is also provided in `utils_improved.py`; to avoid duplicate definitions in a future cleanup, import the class from that module instead.

## Models and explanations

**Random Forest:** 100 estimators, random state `42`, trained on the oversampled training set. Tree SHAP explains stroke-class predictions and supports local force plots and global summary plots.

**RBF-SVM:** A pipeline with `StandardScaler` and `SVC` using an RBF kernel and probability estimates. Kernel SHAP uses a sampled training-data background to approximate local and global feature contributions. Its explanations are computationally more expensive and sensitive to sampling choices.

The data are imbalanced, so accuracy alone is insufficient. The notebook also evaluates recall, precision, F1, balanced accuracy, ROC AUC, average precision, and confusion matrices.

## Reproducibility and GitHub commit

For the exact version submitted with the assignment, record the GitHub repository URL and the commit hash:

```bash
git rev-parse HEAD
```

Use the resulting hash in the assignment report. If this repository is private, grant the lecturer access if required.

## Acknowledgements

The numbered XAI exercise scripts originate from the course teaching materials; the final notebook and improved data-preparation workflow extend the original exercises for the stroke classification and SHAP assignment. Dataset attribution is provided above. Retain original author/licence notices and obtain permission before publicly redistributing teaching materials.
