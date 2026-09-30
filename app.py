"""Streamlit interface for the placement prediction model."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from placement_predictor import placement_probability


ARTIFACT_PATH = Path("artifacts/placement_model.joblib")

st.set_page_config(
    page_title="Placement Intelligence Portal",
    page_icon="🎓",
    layout="wide",
)
st.title("🎓 Student Placement Intelligence")
st.write(
    "Estimate placement readiness and explore model-based areas for development."
)

if not ARTIFACT_PATH.is_file():
    st.warning(
        "No trained model was found. Train one with "
        "`python train_model.py placement_train.csv`, then reload this page."
    )
    st.stop()

artifact = joblib.load(ARTIFACT_PATH)
model = artifact["model"]
feature_columns = artifact["feature_columns"]
medians = artifact["medians"]

with st.form("student_profile"):
    academic, competency = st.columns(2)

    with academic:
        st.subheader("Academic credentials")
        age = st.slider("Age", 18, 30, 21)
        cgpa = st.slider("CGPA", 4.0, 10.0, 8.2, 0.1)
        backlogs = st.number_input("Active backlogs", 0, 8, 0)
        internships = st.number_input("Internships completed", 0, 5, 1)
        projects = st.number_input("Technical projects completed", 0, 8, 3)

    with competency:
        st.subheader("Competency evaluation")
        communication = st.slider("Communication skills (1-10)", 1, 10, 8)
        coding = st.slider("Coding skills (1-10)", 1, 10, 7)
        aptitude = st.slider("Aptitude test score (0-100)", 0, 100, 80)
        soft_skills = st.slider("Soft skills rating (1-5)", 1, 5, 4)

    submitted = st.form_submit_button("Evaluate candidate readiness")

if submitted:
    profile = {
        "Age": age,
        "CGPA": cgpa,
        "Backlogs": backlogs,
        "Internships": internships,
        "Projects": projects,
        "Communication_Skills": communication,
        "Coding_Skills": coding,
        "Aptitude_Test_Score": aptitude,
        "Soft_Skills_Rating": soft_skills,
    }
    probability = placement_probability(model, profile, feature_columns, medians)
    st.metric("Estimated placement probability", f"{probability * 100:.1f}%")

    if probability >= 0.5:
        st.success("The model estimates a higher likelihood of placement.")
    else:
        st.warning("The model estimates a lower likelihood; review the guidance below.")

    st.subheader("Targeted development guidance")
    recommendations = []
    possible_changes = (
        (
            communication < 8,
            "Communication development",
            "Practice group discussions and interview communication; target 8/10.",
            {"Communication_Skills": 8},
        ),
        (
            backlogs > 0,
            "Backlog remediation",
            "Prioritize clearing active backlogs to meet employer eligibility criteria.",
            {"Backlogs": 0},
        ),
        (
            projects < 3,
            "Practical portfolio",
            "Build and deploy projects that demonstrate end-to-end technical skills; target 3 projects.",
            {"Projects": 3},
        ),
    )
    for applicable, title, guidance, change in possible_changes:
        if applicable:
            counterfactual = {**profile, **change}
            changed_probability = placement_probability(
                model, counterfactual, feature_columns, medians
            )
            difference = (changed_probability - probability) * 100
            recommendations.append((difference, title, guidance))

    if recommendations:
        for difference, title, guidance in sorted(
            recommendations, key=lambda item: item[0], reverse=True
        ):
            st.markdown(
                f"- **{title}:** {guidance} "
                f"(Model-estimated change for this profile: {difference:+.1f} percentage points.)"
            )
    else:
        st.info("This profile meets the current development thresholds.")

    st.caption(
        "Counterfactual changes are model estimates based on historical data, "
        "not causal effects or guarantees of an outcome."
    )

    if hasattr(model, "feature_importances_"):
        st.subheader("Top model features")
        importances = pd.Series(
            model.feature_importances_, index=feature_columns
        ).nlargest(10).sort_values()
        st.bar_chart(importances)
