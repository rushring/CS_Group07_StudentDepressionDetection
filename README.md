# 🧠 Student Depression Detection System using ML v1.0
### *Early Detection of Student Depression Using Machine Learning and Stacked Meta-Modeling*

## 📘 Overview
**Student Depression Detection System using ML v1.0** is a machine-learning solution designed to identify students who are at risk of depression.  
The system processes academic, behavioural, and lifestyle-related data to predict mental health conditions using a **stacked Meta-Model** built from five base machine learning models.

The project includes:  
- A full ML training pipeline  
- Stacked ensemble inference  
- A polished **Streamlit web interface**  
- Full support for **Docker & Dev Containers (`app-env`)**

## ⭐ Key Features
- ✔ Depression risk prediction using machine learning  
- ✔ Advanced **stacked meta-model** for superior accuracy  
- ✔ Streamlined data preprocessing pipeline  
- ✔ Interactive **Streamlit app**  
- ✔ Isolated execution through **`app-env` dev container**  
- ✔ Reproducible environment (Docker + VS Code Dev Containers)  
- ✔ Modular and extendable codebase  

## 🏗 System Architecture
```
                         ┌────────────────────┐
                         │  Raw Student Data   │
                         └──────────┬─────────┘
                                    │
                          Preprocessing Pipeline
                                    │
                 ┌────────────── Meta-Model (Stacked) ───────────────┐
                 │                                                    │
     ┌───────────────┬───────────────┬──────────────┬────────────────┬──────────────┐
     │ GradientBoost │ RandomForest  │ DecisionTree │ LogisticReg    │ SVM Model    │
     │ Base Model 1  │ Base Model 2  │ Base Model 3 │ Base Model 4   │ Base Model 5 │
     └──────────────┴───────────────┴──────────────┴────────────────┴──────────────┘
                                   │
                        Meta-Learner (Final Classifier)
                                   │
                            Final Depression Prediction
                                   │
                         Streamlit Web Application (UI)
```

## 📊 Dataset & Features
- Gender  
- Age Group  
- Sleep Duration  
- Dietary Habits  
- Suicidal Thoughts  
- Mental Illness History  
- Label: Depressed (1/0)

## 🔧 Preprocessing Pipeline
- One-Hot Encoding (Gender, Degree, Suicidal Thoughts, Mental Illness History)  
- Ordinal Encoding (Sleep Duration, Dietary Habits, Age Group)  
- Missing value handling  
- Scaling (if needed)  
- Train-test split (80/20)

## 🤖 Meta-Model Architecture
Base Models:  
1. Gradient Boosting  
2. Random Forest  
3. Decision Tree  
4. Logistic Regression  
5. SVM  

Meta-Learner: XGB

## 🧰 Tech Stack
Python, Scikit-Learn, Streamlit, Docker, VS Code Dev Containers, Mamba/Conda

## 🖥 Local Installation Guide
```
git clone <repo>
pip install -r requirements.txt
streamlit run app.py
```

## 🐳 Running in app-env (Dev Container)
### 1. Install:
Docker Desktop + VS Code + Dev Containers extension  

### 2. Open project:
```
code .
```

### 3. Reopen in container:
**Ctrl + Shift + P → Dev Containers: Reopen in Container**

### 4. Install deps inside container:
```
pip install -r requirements.txt
```

### 5. Run:
```
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 6. Access App:
Open: http://localhost:8501


## 🚀 Future Improvements
- Add SHAP/LIME explainability  
- Deploy to cloud  
- Add REST API  

## 👨‍💻 Contributors
- Menura Andrahennedi
- Sivakumar Rushani
- Themiya Dulshan
- Kaveesha Alwis
- Dunith Munasinghe

## 📄 License
MIT License
