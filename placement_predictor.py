"""Shared preprocessing and prediction helpers for the placement app."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "Age": ("age",),
    "CGPA": ("cgpa",),
    "Aptitude_Test_Score": ("aptitude test score", "aptitude", "aptitude score"),
    "Coding_Skills": ("coding skills", "coding"),
    "Projects": ("projects", "project count", "number of projects"),
    "Communication_Skills": ("communication skills", "communication"),
    "Soft_Skills_Rating": ("soft skills rating", "soft skills"),
    "Backlogs": ("backlogs", "active backlogs", "history of backlogs"),
    "Internships": ("internships", "internship count", "internships completed"),
}

ENGINEERED_COLUMNS = (
    "Academic_Aptitude_Balance",
    "Technical_Index",
    "Employability_Rating",
    "Academic_Risk_Score",
)

POSITIVE_LABELS = {"1", "1.0", "yes", "y", "true", "placed", "selected", "eligible"}
NEGATIVE_LABELS = {
    "0",
    "0.0",
    "no",
    "n",
    "false",
    "not placed",
    "not_placed",
    "notplaced",
    "unplaced",
    "rejected",
    "ineligible",
}


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def canonicalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename recognized input columns to the names used by the app."""
    result = frame.copy()
    normalized = {_normalized_name(column): column for column in result.columns}
    renames: dict[Any, str] = {}
    claimed: set[Any] = set()

    for canonical, aliases in COLUMN_ALIASES.items():
        possible_names = dict.fromkeys(
            _normalized_name(name) for name in (canonical, *aliases)
        )
        matches = [
            normalized[name]
            for name in possible_names
            if name in normalized and normalized[name] not in claimed
        ]
        if len(matches) > 1:
            raise ValueError(
                f"Multiple columns match {canonical}: {', '.join(map(str, matches))}."
            )
        if matches:
            source = matches[0]
            claimed.add(source)
            if source != canonical:
                renames[source] = canonical

    return result.rename(columns=renames)


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the composite indicators used by the shared Colab notebook."""
    result = canonicalize_columns(frame)

    if {"CGPA", "Aptitude_Test_Score"} <= set(result.columns):
        result["Academic_Aptitude_Balance"] = (
            result["CGPA"] * (result["Aptitude_Test_Score"] / 100)
        )
    if {"Coding_Skills", "Projects"} <= set(result.columns):
        result["Technical_Index"] = (
            result["Coding_Skills"] * 0.6 + result["Projects"] * 1.5
        )
    if {"Communication_Skills", "Soft_Skills_Rating"} <= set(result.columns):
        result["Employability_Rating"] = (
            result["Communication_Skills"] * 1.2 + result["Soft_Skills_Rating"]
        )
    if {"Backlogs", "CGPA"} <= set(result.columns):
        result["Academic_Risk_Score"] = result["Backlogs"] / (result["CGPA"] + 1e-5)

    return result


def detect_target_column(frame: pd.DataFrame, target: str | None = None) -> str:
    if target:
        matches = [
            column
            for column in frame.columns
            if _normalized_name(column) == _normalized_name(target)
        ]
        if not matches:
            raise ValueError(f"Target column {target!r} was not found in the CSV.")
        return matches[0]

    candidates = [
        column
        for column in frame.columns
        if any(word in str(column).lower() for word in ("placed", "placement", "status"))
        and not any(word in str(column).lower() for word in ("rating", "score"))
    ]
    if not candidates:
        raise ValueError(
            "Could not identify the placement outcome column. Pass its name with --target."
        )
    if len(candidates) > 1:
        raise ValueError(
            "More than one possible target column was found "
            f"({', '.join(map(str, candidates))}); pass the intended one with --target."
        )
    return candidates[0]


def encode_target(values: pd.Series) -> pd.Series:
    """Convert common binary placement labels to 0/1 and reject ambiguity."""
    if values.isna().any():
        raise ValueError("The target column contains missing values.")

    if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
        numeric_values = pd.to_numeric(values)
        if not numeric_values.isin([0, 1]).all():
            raise ValueError("Numeric target values must be exactly 0 or 1.")
        numeric = numeric_values.astype(int)
    else:
        labels = values.astype(str).str.strip().str.lower()
        unknown = sorted(set(labels) - POSITIVE_LABELS - NEGATIVE_LABELS)
        if unknown:
            raise ValueError(
                "The target must be binary placement labels (for example, "
                f"Placed/Not Placed or 1/0); unrecognized values: {unknown[:8]}."
            )
        numeric = labels.map(
            lambda label: 1 if label in POSITIVE_LABELS else 0
        ).astype(int)

    if set(numeric.unique()) != {0, 1}:
        raise ValueError("The target column must contain both binary classes 0 and 1.")
    return numeric


def model_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Create numeric model inputs with the same one-hot encoding as training."""
    prepared = engineer_features(frame)
    prepared = prepared.drop(
        columns=[
            column
            for column in prepared.columns
            if _normalized_name(column) in {"studentid", "studentidentifier"}
        ],
        errors="ignore",
    )
    matrix = pd.get_dummies(prepared, drop_first=True, dtype=float)
    return matrix.apply(pd.to_numeric, errors="raise")


def prepare_model_input(
    student_data: Mapping[str, Any],
    feature_columns: Sequence[str],
    medians: Mapping[str, float],
) -> pd.DataFrame:
    matrix = model_matrix(pd.DataFrame([dict(student_data)]))
    matrix = matrix.reindex(columns=list(feature_columns), fill_value=0)
    return matrix.fillna(pd.Series(medians)).fillna(0)


def placement_probability(
    model: Any,
    student_data: Mapping[str, Any],
    feature_columns: Sequence[str],
    medians: Mapping[str, float],
) -> float:
    model_input = prepare_model_input(student_data, feature_columns, medians)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The trained model does not contain the placed (1) class.")
    return float(model.predict_proba(model_input)[0][classes.index(1)])
