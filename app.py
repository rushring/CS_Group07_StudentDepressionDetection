import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------------------------------
# 1. Load models (UPDATE THESE PATHS!)
# ---------------------------------------------------
DT_PATH  = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/dunith_decision_tree/decision_tree_model.joblib"
RF_PATH  = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/fc211009_Themiya_random_forrest/random_forest_student_depression.joblib"
SVM_PATH = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/fc211011_kaveesha_svm_model/best_svm_model.joblib"
GB_PATH  = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/menura-gradient_boosting_classifier/gbc_model.joblib"
LR_PATH  = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/Rushani_Logistic_Regression/final_logistic_model.joblib"
META_PATH = "/workspaces/CS_Group07_StudentDepressionDetection/notebooks/meta_model/final_meta_model.joblib"

@st.cache_resource
def load_models():
    dt_model  = joblib.load(DT_PATH)
    rf_model  = joblib.load(RF_PATH)
    svm_model = joblib.load(SVM_PATH)
    gb_model  = joblib.load(GB_PATH)
    lr_model  = joblib.load(LR_PATH)
    meta_model = joblib.load(META_PATH)
    return dt_model, rf_model, svm_model, gb_model, lr_model, meta_model

dt_model, rf_model, svm_model, gb_model, lr_model, meta_model = load_models()


# ---------------------------------------------------
# CSS — UI Enhancement
# ---------------------------------------------------
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
        body {
            background-color: #f4f7fa;
        }
        .main-title {
            font-size: 38px;
            font-weight: 700;
            text-align: center;
            color: #2b4eff;
            margin-top: -20px;
        }
        .subtext {
            text-align: center;
            font-size: 17px;
            color: #444;
            margin-bottom: 30px;
        }
        .card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        }
        .predict-btn button {
            width: 100%;
            padding: 15px;
            border-radius: 12px !important;
            font-size: 18px;
        }
        .result-card {
            background: #ffffff;
            padding: 25px;
            border-radius: 15px;
            font-size: 18px;
            text-align: center;
            margin-bottom: 15px;
        }
        .model-result {
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 12px;
            border-left: 5px solid;
        }
        .positive {
            background: #8B0000;
            border-left-color: #c62828;
        }
        .negative {
            background: #008000;
            border-left-color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🧠 Student Depression Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtext'>Fill in the details below to get predictions from all 6 models</p>", unsafe_allow_html=True)


# ---------------------------------------------------
# 2. Define final 46 feature columns (must match training)
# ---------------------------------------------------
MODEL_COLUMNS = [
    "Age", "Academic_Pressure", "CGPA", "Study_Satisfaction", "Study_Hours",
    "Financial_Stress", "Gender_Male",
    "Sleep_Duration_7-8 hours", "Sleep_Duration_Less than 5 hours",
    "Sleep_Duration_More than 8 hours", "Sleep_Duration_Others",
    "Dietary_Habits_Moderate", "Dietary_Habits_Others", "Dietary_Habits_Unhealthy",
    "Degree_B.Com", "Degree_B.Ed", "Degree_B.Pharm", "Degree_B.Tech", "Degree_BA",
    "Degree_BBA", "Degree_BCA", "Degree_BE", "Degree_BHM", "Degree_BSc",
    "Degree_Class 12", "Degree_LLB", "Degree_LLM", "Degree_M.Com", "Degree_M.Ed",
    "Degree_M.Pharm", "Degree_M.Tech", "Degree_MA", "Degree_MBA", "Degree_MBBS",
    "Degree_MCA", "Degree_MD", "Degree_ME", "Degree_MHM", "Degree_MSc",
    "Degree_Others", "Degree_PhD",
    "Suicidal Thoughts_Yes", "Mental Illness History_Yes",
    "Age_Group_Senior", "Age_Group_Teen", "Age_Group_Young Adult"
]

# The meta-model's feature names
META_COLUMNS = [
    'DecisionTree',
    'RandomForest',
    'SVM',
    'GradientBoosting',
    'LogisticRegression',
    'DecisionTree_abs',
    'RandomForest_abs',
    'SVM_abs',
    'GradientBoosting_abs',
    'LogisticRegression_abs',
    'entropy',
    'mean_proba',
    'std_proba'
]

# Model information
MODEL_INFO = {
    "Decision Tree": {"model": dt_model, "icon": "🌳"},
    "Random Forest": {"model": rf_model, "icon": "🌲"},
    "SVM": {"model": svm_model, "icon": "📊"},
    "Gradient Boosting": {"model": gb_model, "icon": "🚀"},
    "Logistic Regression": {"model": lr_model, "icon": "📈"},
    "Meta Model": {"model": meta_model, "icon": "🎯"}
}


# ---------------------------------------------------
# 3. Streamlit UI
# ---------------------------------------------------
st.markdown("### 📝 Personal & Study Information")

col1, col2 = st.columns(2)

with col1:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        age = st.number_input("🎯 Age", min_value=10, max_value=80, value=20)
        academic_pressure = st.slider("📚 Academic Pressure (0–10)", 0, 10, 5)
        cgpa = st.slider("🎓 CGPA (0–4)", 0.0, 4.0, 3.0, step=0.1)
        study_satisfaction = st.slider("😊 Study Satisfaction (0–10)", 0, 10, 5)
        study_hours = st.slider("⏳ Study Hours Per Day", 0, 15, 4)
        financial_stress = st.slider("💰 Financial Stress (0–10)", 0, 10, 5)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        gender = st.selectbox("🚻 Gender", ["Male", "Female"])
        sleep_duration = st.selectbox("😴 Sleep Duration", ["7-8 hours", "Less than 5 hours", "More than 8 hours", "Others"])
        diet = st.selectbox("🍽️ Dietary Habits", ["Moderate", "Others", "Unhealthy"])
        degree = st.selectbox("🎓 Degree Program", [
            "B.Com", "B.Ed", "B.Pharm", "B.Tech", "BA", "BBA", "BCA", "BE", "BHM",
            "BSc", "Class 12", "LLB", "LLM", "M.Com", "M.Ed", "M.Pharm", "M.Tech",
            "MA", "MBA", "MBBS", "MCA", "MD", "ME", "MHM", "MSc", "Others", "PhD"
        ])
        suicidal = st.selectbox("⚠️ Suicidal Thoughts", ["No", "Yes"])
        mental_history = st.selectbox("🧩 Mental Illness History", ["No", "Yes"])
        age_group = st.selectbox("👤 Age Group", ["Teen", "Young Adult", "Senior"])
        st.markdown("</div>", unsafe_allow_html=True)


st.markdown("---")

st.markdown("### 🤖 Select Models for Prediction")
selected_models = st.multiselect(
    "Choose which models to use for prediction:",
    list(MODEL_INFO.keys()),
    default=list(MODEL_INFO.keys())
)

st.markdown("---")


# ---------------------------------------------------
# 4. Build 46-feature row from user input
# ---------------------------------------------------
def build_feature_row():
    data = dict.fromkeys(MODEL_COLUMNS, 0.0)

    # numeric
    data["Age"] = float(age)
    data["Academic_Pressure"] = float(academic_pressure)
    data["CGPA"] = float(cgpa)
    data["Study_Satisfaction"] = float(study_satisfaction)
    data["Study_Hours"] = float(study_hours)
    data["Financial_Stress"] = float(financial_stress)

    # gender
    if gender == "Male":
        data["Gender_Male"] = 1.0

    # sleep duration
    data[f"Sleep_Duration_{sleep_duration}"] = 1.0

    # diet
    data[f"Dietary_Habits_{diet}"] = 1.0

    # degree
    data[f"Degree_{degree}"] = 1.0

    # suicidal & mental history
    if suicidal == "Yes":
        data["Suicidal Thoughts_Yes"] = 1.0
    if mental_history == "Yes":
        data["Mental Illness History_Yes"] = 1.0

    # age group
    data[f"Age_Group_{age_group}"] = 1.0

    return pd.DataFrame([data])

# ---------------------------------------------------
# 5. From 46 features → base model outputs → meta features
# ---------------------------------------------------
def build_meta_features(x_user: pd.DataFrame) -> pd.DataFrame:
    # Compute probabilities
    def safe_proba(model, X):
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(X)[0, 1])
        if hasattr(model, "decision_function"):
            val = float(model.decision_function(X)[0])
            return 1 / (1 + np.exp(-val))
        return float(model.predict(X)[0])

    # Base model outputs
    p_dt = safe_proba(dt_model, x_user)
    p_rf = safe_proba(rf_model, x_user)
    p_svm = safe_proba(svm_model, x_user)
    p_gb = safe_proba(gb_model, x_user)
    p_lr = safe_proba(lr_model, x_user)

    probs = np.array([p_dt, p_rf, p_svm, p_gb, p_lr])

    # Extra features
    eps = 1e-15
    entropy = -np.mean(probs * np.log(probs + eps) + (1 - probs) * np.log(1 - probs + eps))
    mean_proba = probs.mean()
    std_proba = probs.std()

    # Create DataFrame IN CORRECT ORDER
    meta_row = [
        p_dt,
        p_rf,
        p_svm,
        p_gb,
        p_lr,
        abs(p_dt),
        abs(p_rf),
        abs(p_svm),
        abs(p_gb),
        abs(p_lr),
        entropy,
        mean_proba,
        std_proba
    ]

    return pd.DataFrame([meta_row], columns=META_COLUMNS)

# ---------------------------------------------------
# Get prediction from a specific model
# ---------------------------------------------------
def get_model_prediction(model, x_user: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(x_user)[0, 1])
    elif hasattr(model, "decision_function"):
        val = float(model.decision_function(x_user)[0])
        proba = 1 / (1 + np.exp(-val))
    else:
        proba = float(model.predict(x_user)[0])
    
    return proba


# ---------------------------------------------------
# Predict Button + Result UI
# ---------------------------------------------------
st.markdown("### 🔍 Prediction")

button_container = st.container()
with button_container:
    st.markdown("<div class='predict-btn'>", unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict Depression")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    if not selected_models:
        st.warning("⚠️ Please select at least one model!")
    else:
        X_user = build_feature_row()
        
        st.markdown("### 📊 Model Predictions")
        
        # Display results in columns
        cols = st.columns(min(3, len(selected_models)))
        
        for idx, model_name in enumerate(selected_models):
            col = cols[idx % len(cols)]
            
            model_obj = MODEL_INFO[model_name]["model"]
            icon = MODEL_INFO[model_name]["icon"]
            
            # Get meta features for meta model
            if model_name == "Meta Model":
                meta_features = build_meta_features(X_user)
                proba = float(meta_model.predict_proba(meta_features)[0, 1])
            else:
                proba = get_model_prediction(model_obj, X_user)
            
            pred = 1 if proba >= 0.5 else 0
            status = "🚨 Depression" if pred == 1 else "✅ No Depression"
            bg_class = "positive" if pred == 1 else "negative"
            
            with col:
                st.markdown(f"""
                    <div class='model-result {bg_class}'>
                        <h4>{icon} {model_name}</h4>
                        <p><b>Probability:</b> {proba:.1%}</p>
                        <p><b>Prediction:</b> {status}</p>
                    </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Summary section
        st.markdown("### 📈 Summary")
        meta_features = build_meta_features(X_user)
        meta_proba = float(meta_model.predict_proba(meta_features)[0, 1])
        
        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            st.metric("Meta Model Probability", f"{meta_proba:.1%}")
        with summary_col2:
            st.metric("Meta Model Prediction", "🚨 Depression Likely" if meta_proba >= 0.5 else "✅ No Depression")