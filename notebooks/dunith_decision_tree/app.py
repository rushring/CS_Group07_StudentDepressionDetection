import streamlit as st
import pickle
import numpy as np
import pandas as pd

#  Load the trained model
model = pickle.load(open("decision_tree_model.pkl", "rb"))  

st.set_page_config(page_title="Student Depression Prediction", page_icon="🧠", layout="centered")

st.title("🧠 Student Depression Prediction App")
st.write("This app uses a Decision Tree model to predict whether a student is likely to have depression based on lifestyle and academic factors.")

#  Collect user inputs
st.subheader("Enter Student Details:")

age = st.number_input("Age", 10, 50, 21)
academic_pressure = st.slider("Academic Pressure (1–10)", 1, 10, 5)
study_satisfaction = st.slider("Study Satisfaction (1–10)", 1, 10, 5)
cgpa = st.number_input("CGPA", 0.0, 4.0, 3.0)
study_hours = st.slider("Work/Study Hours per Day", 0, 15, 6)
financial_stress = st.slider("Financial Stress (1–10)", 1, 10, 5)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])
profession = st.selectbox("Profession", ["Student", "Employed", "Unemployed"])
sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Unhealthy"])
degree = st.selectbox("Degree", ["Undergraduate", "Postgraduate", "Other"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
mental_illness_history = st.selectbox("Family History of Mental Illness?", ["No", "Yes"])

if age <= 18:
    age_group = "Teen"
elif age <= 25:
    age_group = "Young Adult"
elif age <= 35:
    age_group = "Adult"
else:
    age_group = "Senior"

#  Encode categorical variables (ensure the encoding matches your LabelEncoder from training)
mapping = {
    "Male": 1, "Female": 0, "Other": 2,
    "Student": 2, "Employed": 0, "Unemployed": 1,
    "Less than 5 hours": 0, "5-6 hours": 1, "7-8 hours": 2, "More than 8 hours": 3,
    "Healthy": 0, "Unhealthy": 1,
    "Undergraduate": 0, "Postgraduate": 1, "Other": 2,
    "No": 0, "Yes": 1,
    "Teen": 0, "Young Adult": 1, "Adult": 2, "Senior": 3 
}

#  Feature creation (must match training structure)
overall_stress = academic_pressure + study_hours + financial_stress

input_data = np.array([
    age,
    academic_pressure,
    study_satisfaction,
    cgpa,
    study_hours,
    financial_stress,
    mapping[gender],
    mapping[profession],
    mapping[sleep_duration],
    mapping[dietary_habits],
    mapping[degree],
    mapping[suicidal_thoughts],
    mapping[mental_illness_history],
    mapping[age_group],  
    overall_stress
]).reshape(1, -1)

# Predict button
if st.button("🔍 Predict Depression"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts the student **might be suffering from depression.** Please consider professional help.")
    else:
        st.success("✅ The model predicts the student is **not likely to be depressed.**")

st.markdown("---")
st.caption("Developed by Dunith |  Decision Tree Classifier ")
