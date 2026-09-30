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

## Run the app

```powershell
streamlit run app.py
```

The app displays a probability estimate and compares hypothetical profile
changes for communication skills, backlogs, and projects. These are model-based
associations, not guarantees or causal estimates. Predictions are only as
reliable as the dataset used to train the model.
