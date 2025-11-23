import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import json

# ===================== CSS =====================
st.markdown("""
    <style>
        .block-container {
            max-width: 100% !important;
            padding-left: 5rem !important;
            padding-right: 5rem !important;
        }

        .css-1kyxreq {
            gap: 3rem !important;
        }
        .main-title {
            text-align: center;
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 20px;
            color: #2E86C1;
        }
        .sub-title {
            text-align: center;
            font-size: 18px;
            color: #555;
            margin-bottom: 30px;
        }
        .result-card {
            background-color: #F2F4F4;
            padding: 20px;
            border-radius: 12px;
            border-left: 5px solid #2E86C1;
            margin-top: 20px;
        }
        .probability-text {
            font-size: 26px;
            font-weight: 700;
            color: #1A5276;
        }
            
        div.stButton > button {
            height: 50px !important;
            font-size: 20px !important;
            font-weight: 700 !important;
            background-color: #0D0D0D !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid white !important;
        }

    </style>
""", unsafe_allow_html=True)


st.set_page_config(page_title="Student Depression Prediction", layout="centered")


# --------------- Load saved models ---------------
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

# --------------- Prediction function ---------------
def run_prediction(input_dict):
    df = pd.DataFrame([input_dict])

    # Base models
    dt_pred = dt_model.predict_proba(df)[0][1]
    rf_pred = rf_model.predict_proba(df)[0][1]
    svm_pred = svm_model.predict_proba(df)[0][1]
    gb_pred = gb_model.predict_proba(df)[0][1]
    lr_pred = lr_model.predict_proba(df)[0][1]

    base_outputs = {
        "DecisionTree": float(dt_pred),
        "RandomForest": float(rf_pred),
        "SVM": float(svm_pred),
        "GradientBoosting": float(gb_pred),
        "LogisticRegression": float(lr_pred)
    }

    # Meta model
    meta_input = pd.DataFrame([base_outputs])
    meta_pred = best_meta_model.predict_proba(meta_input)[0][1]

    return meta_pred, base_outputs

# # --------------- Input fields ---------------
# st.markdown("<h2 class='main-title'>Student Depression Prediction System</h2>", unsafe_allow_html=True)
# st.markdown("<p class='sub-title'>Enter student information below and click Predict</p>", unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age", 15, 50, 25)
#     academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
#     cgpa = st.number_input("CGPA", 0.0, 4.0, 3.0, step=0.01)
#     study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
#     study_hours = st.slider("Study Hours", 0, 16, 6)
#     financial_stress = st.slider("Financial Stress", 1, 5, 3)

# with col2:
#     gender = st.selectbox("Gender", ["Male", "Female"])
#     sleep_duration = st.selectbox(
#         "Sleep Duration",
#         ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
#         index=2
#     )
#     dietary_habits = st.selectbox(
#         "Dietary Habits",
#         ["Unhealthy", "Moderate", "Healthy"],
#         index=2
#     )
#     degree = st.selectbox("Degree", ["Undergraduate", "Postgraduate"], index=0)
#     suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["Yes", "No"], index=0)
#     mental_illness_history = st.selectbox("Mental Illness History", ["Yes", "No"], index=0)


# # ---------------- Create input row ----------------
# input_dict = {
#     "Age": age,
#     "Academic_Pressure": academic_pressure,
#     "CGPA": cgpa,
#     "Study_Satisfaction": study_satisfaction,
#     "Study_Hours": study_hours,
#     "Financial_Stress": financial_stress,
#     "Gender": gender,
#     "Sleep_Duration": sleep_duration,
#     "Dietary_Habits": dietary_habits,
#     "Degree": degree,
#     "Suicidal_Thoughts": suicidal_thoughts,
#     "Mental_Illness_History": mental_illness_history
# }

# input_df = pd.DataFrame([input_dict])


# # --------------- PREDICTIONS ---------------
# st.markdown("<div class='center-button'>", unsafe_allow_html=True)
# predict_clicked = st.button("Predict")
# st.markdown("</div>", unsafe_allow_html=True)

# if predict_clicked:
#     final_pred, base_outputs = run_prediction(input_dict)
    
#     if final_pred > 0.5:
#         st.markdown(f"""
#             <div style="padding: 20px; border-radius: 10px; background-color:#ffcccc;">
#                 <h2 style="color:#cc0000;">🚨 High Risk of Depression</h2>
#                 <h3 style="color:#000000; font-size:20px;">Probability of being depressed: {final_pred:.4f}</h3>
#             </div>
#         """, unsafe_allow_html=True)
#     else:
#         st.markdown(f"""
#             <div style="padding: 20px; border-radius: 10px; background-color:#ccffdd;">
#                 <h2 style="color:#006600;">✅ Low Risk of Depression</h2>
#                 <h3 style="color:#000000; font-size:20px;">Probability of being depressed: {final_pred:.4f}</h3>
#             </div>
#         """, unsafe_allow_html=True)

#     st.markdown("<br>", unsafe_allow_html=True)
#     with st.expander("🔍 Show Base Model's Individual Predictions"):
#         st.json(base_outputs)


# ------------------------------------------------------------
#                     MAIN TWO-COLUMN LAYOUT
# ------------------------------------------------------------
left, right = st.columns([7, 6], gap="large")  # 2/3 left, 1/3 right


with left:

    # Title
    st.markdown("<h1 class='main-title'>Student Depression Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title' style='font-size:25px;'>Enter student information below and click Predict</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 15, 50, 25)
        academic_pressure = st.slider("Academic Pressure", 1, 5, 3)
        cgpa = st.number_input("CGPA", 0.0, 4.0, 3.0, step=0.01)
        study_satisfaction = st.slider("Study Satisfaction", 1, 5, 3)
        study_hours = st.slider("Study Hours", 0, 16, 6)
        financial_stress = st.slider("Financial Stress", 1, 5, 3)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"],index=0)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
            index=1
        )
        dietary_habits = st.selectbox(
            "Dietary Habits",
            ["Unhealthy", "Moderate", "Healthy"],
            index=2
        )
        degree = st.selectbox("Degree", ["Undergraduate", "Postgraduate"], index=0)
        suicidal_thoughts = st.selectbox("Suicidal Thoughts", ["Yes", "No"], index=1)
        mental_illness_history = st.selectbox("Mental Illness History", ["Yes", "No"], index=1)

    # Prepare input row
    input_dict = {
        "Age": age,
        "Academic_Pressure": academic_pressure,
        "CGPA": cgpa,
        "Study_Satisfaction": study_satisfaction,
        "Study_Hours": study_hours,
        "Financial_Stress": financial_stress,
        "Gender": gender,
        "Sleep_Duration": sleep_duration,
        "Dietary_Habits": dietary_habits,
        "Degree": degree,
        "Suicidal_Thoughts": suicidal_thoughts,
        "Mental_Illness_History": mental_illness_history
    }


    # Predict button
    full_width_container = st.container()

    with full_width_container:
        predict_clicked = st.button("PREDICT",use_container_width=True)

    if predict_clicked:
        final_pred, base_outputs = run_prediction(input_dict)

        # Risk message
        if final_pred > 0.5:
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color:#ffcccc;">
                    <h2 style="color:#cc0000;">🚨 High Risk of Depression</h2>
                    <h3 style="color:#000000; font-size:20px;">Probability of being depressed: {final_pred:.4f}</h3>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="padding: 20px; border-radius: 10px; background-color:#ccffdd;">
                    <h2 style="color:#006600;">✅ Low Risk of Depression</h2>
                    <h3 style="color:#000000; font-size:20px;">Probability of being depressed: {final_pred:.4f}</h3>
                </div>
            """, unsafe_allow_html=True)

        # Base models result expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("🔍 Show Base Model's Individual Predictions"):
            st.json(base_outputs)


# ------------------------------------------------------------
#                     RIGHT SIDE IMAGE
# ------------------------------------------------------------
with right:
    st.image("/workspaces/CS_Group07_StudentDepressionDetection/Data/image.jpg", use_container_width=True)