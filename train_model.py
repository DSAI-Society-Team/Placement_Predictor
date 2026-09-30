"""Train and compare placement classifiers from a labeled CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from placement_predictor import (
    canonicalize_columns,
    detect_target_column,
    encode_target,
    model_matrix,
)


def train(data_path: Path, target_name: str | None, output_path: Path) -> None:
    if not data_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {data_path}")

    frame = canonicalize_columns(pd.read_csv(data_path))
    target = detect_target_column(frame, target_name)
    labels = encode_target(frame[target])
    raw_features = frame.drop(columns=[target])
    raw_features = raw_features.drop(
        columns=[
            column
            for column in raw_features.columns
            if str(column).strip().lower().replace("_", "").replace(" ", "")
            in {"studentid", "studentidentifier"}
        ],
        errors="ignore",
    )
    if raw_features.shape[1] == 0:
        raise ValueError("The CSV has no usable feature columns.")

    try:
        raw_train, raw_test, y_train, y_test = train_test_split(
            raw_features,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
    except ValueError as error:
        raise ValueError(
            "Could not create a stratified 80/20 split. Include enough examples "
            "of both placement outcomes, or provide a larger dataset."
        ) from error

    x_train = model_matrix(raw_train)
    x_test = model_matrix(raw_test).reindex(columns=x_train.columns, fill_value=0)
    medians = x_train.median().fillna(0)
    x_train = x_train.fillna(medians)
    x_test = x_test.fillna(medians)
    if x_train.shape[1] == 0:
        raise ValueError("The CSV has no usable feature columns after preprocessing.")

    models = {
        "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000),
        "Random Forest (Primary)": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "LightGBM (Gradient Boosting)": LGBMClassifier(
            n_estimators=100, random_state=42, verbosity=-1
        ),
    }

    results: list[dict[str, str | float]] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
        predictions = (probabilities >= 0.5).astype(int)
        results.append(
            {
                "Model": name,
                "Accuracy": round(accuracy_score(y_test, predictions), 4),
                "F1-Score": round(f1_score(y_test, predictions, zero_division=0), 4),
                "ROC-AUC": round(roc_auc_score(y_test, probabilities), 4),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": models["Random Forest (Primary)"],
            "feature_columns": x_train.columns.tolist(),
            "medians": medians.to_dict(),
            "target_column": target,
            "metrics": results,
        },
        output_path,
    )

    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nSaved Random Forest model and preprocessing schema to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to the placement training CSV")
    parser.add_argument(
        "--target",
        help="Name of the binary placement outcome column (auto-detected by default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/placement_model.joblib"),
        help="Model artifact path (default: artifacts/placement_model.joblib)",
    )
    args = parser.parse_args()
    train(args.csv, args.target, args.output)


if __name__ == "__main__":
    main()
