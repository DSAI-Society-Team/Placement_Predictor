# Placement Predictor

A Streamlit app that estimates a student's placement probability from a trained
classification model and suggests areas to work on. The training workflow is
adapted from the shared Colab notebook.

## Requirements

- Python 3.10 or later
- A labeled CSV dataset (the dataset is not included in this repository)

## Set up and train

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py placement_train.csv
```

The CSV must contain a binary placement outcome column. The training script
detects columns whose names contain `placed`, `placement`, or `status`; if more
than one column matches, specify the intended one with `--target`.

The script recognizes common variants of the notebook's input columns, such as
`Aptitude`/`Aptitude_Test_Score`, `Project_Count`/`Projects`, and
`Active_Backlogs`/`Backlogs`. Missing values in numeric inputs are imputed using
the training split. The primary Random Forest model and its feature schema are
saved to `artifacts/placement_model.joblib`.

The model comparison reports both the stratified holdout metrics and five-fold
cross-validation metrics computed only within the training partition. Use the
held-out split as the final comparison; cross-validation is for model stability
and selection, not a substitute for an independent external cohort. The
logistic-regression baseline standardizes features and uses `C=10`, selected
with training-only cross-validation to improve its ROC-AUC and convergence.

## Run the app

```powershell
streamlit run app.py
```

The app displays a probability estimate and compares hypothetical profile
changes for communication skills, backlogs, and projects. These are model-based
associations, not guarantees or causal estimates. Predictions are only as
reliable as the dataset used to train the model.

## Standalone ROC-AUC report

Open `index.html` in a browser to inspect the reproducible stratified 80/20
holdout evaluation without running Streamlit or a server. It embeds the 9,000
held-out actual labels and predicted probabilities for Logistic Regression,
Random Forest, and LightGBM, then calculates ROC-AUC, average precision,
thresholded metrics, confusion matrices, and ROC points in the browser.

Use **Download all held-out scores CSV** to give another AI or tool the actual
labels and per-model probabilities needed to independently recalculate every
metric. **Download this model's ROC points CSV** exports the curve thresholds
and rates. The original dataset and student attributes are not included.
