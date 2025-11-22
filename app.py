import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import json

st.set_page_config(page_title="Student Depression Prediction", layout="centered")

# -----------------------
# Load saved models
# -----------------------
best_meta_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/meta_model/best_meta_model.joblib")

dt_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/dunith_decision_tree/decision_tree_model.joblib")
rf_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/fc211009_Themiya_random_forrest/random_forest_student_depression.joblib")
svm_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/fc211011_kaveesha_svm_model/best_svm_model.joblib")
gb_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/menura-gradient_boosting_classifier/gbc_model.joblib")
lr_model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/Rushani_Logistic_Regression/final_logistic_model.joblib")

base_models = {
    'DecisionTree': dt_model,
    'RandomForest': rf_model,
    'SVM': svm_model,
    'GradientBoosting': gb_model,
    'LogisticRegression': lr_model
}

st.title("🧠 Student Depression Prediction System")
st.write("Provide the student's information to predict depression risk.")

# -----------------------
# Input fields
# -----------------------
st.subheader("🔹 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 15, 40, 22)
    Academic_Pressure = st.slider("Academic Pressure (1–5)", 1, 5, 3)
    CGPA = st.slider("CGPA (0–4)", 0.0, 4.0, 3.0)
    Study_Satisfaction = st.slider("Study Satisfaction (1–5)", 1, 5, 3)
    Study_Hours = st.slider("Daily Study Hours", 0, 12, 3)
    Financial_Stress = st.slider("Financial Stress (1–5)", 1, 5, 3)

with col2:
    Gender = st.selectbox("Gender", ["Male", "Female"],index=0)
    Sleep_Duration = st.selectbox("Sleep Duration", 
                                  ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],index=2)

    Dietary_Habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"],index=2)
    Degree = st.selectbox("Degree", ["Undergraduate", "Postgraduate"],index=0)
    Suicidal_Thoughts = st.selectbox("Suicidal Thoughts", ["Yes", "No"],index=1)
    Mental_Illness_History = st.selectbox("Mental Illness History", ["Yes", "No"],index=1)

# -----------------------
# Create input row
# -----------------------
input_dict = {
    "Age": Age,
    "Academic_Pressure": Academic_Pressure,
    "CGPA": CGPA,
    "Study_Satisfaction": Study_Satisfaction,
    "Study_Hours": Study_Hours,
    "Financial_Stress": Financial_Stress,
    "Gender": Gender,
    "Sleep_Duration": Sleep_Duration,
    "Dietary_Habits": Dietary_Habits,
    "Degree": Degree,
    "Suicidal_Thoughts": Suicidal_Thoughts,
    "Mental_Illness_History": Mental_Illness_History
}

input_df = pd.DataFrame([input_dict])


# -----------------------
# META-MODEL PREDICTION
# -----------------------
if st.button("🔍 Predict Depression Status"):

    st.subheader("📌 Model Results")
    
    base_preds = []

    for name, model in base_models.items():
        prob = model.predict_proba(input_df)[0][1]
        base_preds.append(prob)

        st.write(f"**{name} Probability:** {prob:.4f}")

    meta_input = np.array(base_preds).reshape(1, -1)
    final_prob = best_meta_model.predict_proba(meta_input)[0][1]

    st.markdown("---")
    st.subheader("🎯 Final Meta-Model Prediction")

    if final_prob >= 0.5:
        st.error(f"🚨 **Depressed** (Probability: {final_prob:.4f})")
    else:
        st.success(f"✅ **Not Depressed** (Probability: {final_prob:.4f})")
