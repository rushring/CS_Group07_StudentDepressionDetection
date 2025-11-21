import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------
# Load Model
# ---------------------------
model = joblib.load("/workspaces/CS_Group07_StudentDepressionDetection/notebooks/menura-gradient_boosting_classifier/gbc_model.joblib")

st.set_page_config(page_title="Student Depression Detection", layout="wide")

# ---------------------------
# Beautiful Header
# ---------------------------
st.markdown("""
    <h1 style="text-align:center; color:#4A90E2;">
        Student Depression Detection System
    </h1>
    <p style="text-align:center; font-size:18px;">
        Enter student details below to predict depression probability.
    </p>
""", unsafe_allow_html=True)

st.write("")  


# ---------------------------
# Sidebar Styling
# ---------------------------
st.sidebar.header("About")
st.sidebar.info(
    "This app uses a Gradient Boosting Machine (GBM) model trained on student well-being data "
    "to identify depression likelihood. All predictions are probabilistic."
)


# ---------------------------
# Input Form
# ---------------------------
st.subheader("📝 Enter Student Information")

col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=15, max_value=60, step=1)
    Academic_Pressure = st.slider("Academic Pressure (1-5)", 1, 5, 3)
    CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
    Study_Satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)
    Sleep_Duration = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
    )

with col2:
    Dietary_Habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    Degree = st.selectbox("Degree", ["BA", "BCA", "BSc", "B.Pharm", "M.Tech"])
    Suicidal_Thoughts = st.selectbox("Suicidal Thoughts", ["Yes", "No"])
    Study_Hours = st.number_input("Daily Study Hours", min_value=0, max_value=15, step=1)
    Financial_Stress = st.slider("Financial Stress (1-5)", 1, 5, 2)
    Mental_Illness_History = st.selectbox("Mental Illness History", ["Yes", "No"])

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Predict"):
    
    # Create Input DataFrame (Matches training features)
    input_data = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "Academic_Pressure": Academic_Pressure,
        "CGPA": CGPA,
        "Study_Satisfaction": Study_Satisfaction,
        "Sleep_Duration": Sleep_Duration,
        "Dietary_Habits": Dietary_Habits,
        "Degree": Degree,
        "Suicidal_Thoughts": Suicidal_Thoughts,
        "Study_Hours": Study_Hours,
        "Financial_Stress": Financial_Stress,
        "Mental_Illness_History": Mental_Illness_History,
    }])

    # Predict
    proba = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.write("")

    # ---------------------------
    # Display Result
    # ---------------------------
    if pred == 1:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color:#ffcccc;">
                <h2 style="color:#cc0000;">🚨 High Risk of Depression</h2>
                <h3>Probability: {proba:.2f}</h3>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color:#ccffdd;">
                <h2 style="color:#006600;">✅ Low Risk of Depression</h2>
                <h3>Probability: {proba:.2f}</h3>
            </div>
        """, unsafe_allow_html=True)


# ---------------------------
# Footer
# ---------------------------
st.write("")
st.markdown("""
    <hr>
    <p style="text-align:center;">
        Developed with ❤️ using Streamlit & Gradient Boosting Classifier.
    </p>
""", unsafe_allow_html=True)
