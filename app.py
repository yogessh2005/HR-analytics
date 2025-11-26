import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource
def load_model():
    model = joblib.load("./output/best_rf.pkl")  # change to your file path if needed
    return model

model = load_model()

st.title("🧠 IBM HR Analytics – Employee Attrition Prediction App")
st.write("""
This app predicts whether an employee is likely to **leave** or **stay**  
based on key HR attributes such as job satisfaction, income, and experience.
""")


st.sidebar.header("Enter Employee Details")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 60, 30)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    job_role = st.sidebar.selectbox(
        "Job Role",
        [
            "Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director",
            "Healthcare Representative", "Manager", "Sales Representative", "Human Resources", "Research Director"
        ]
    )
    monthly_income = st.sidebar.number_input("Monthly Income", 1000, 20000, 5000, step=500)
    years_at_company = st.sidebar.slider("Years at Company", 0, 40, 5)
    overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
    job_satisfaction = st.sidebar.slider("Job Satisfaction (1–4)", 1, 4, 3)
    work_life_balance = st.sidebar.slider("Work-Life Balance (1–4)", 1, 4, 3)
    environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1–4)", 1, 4, 3)
    years_since_last_promotion = st.sidebar.slider("Years Since Last Promotion", 0, 15, 2)
    performance_rating = st.sidebar.slider("Performance Rating (1–4)", 1, 4, 3)

    data = {
        "Age": [age],
        "Gender": [gender],
        "Department": [department],
        "JobRole": [job_role],
        "MonthlyIncome": [monthly_income],
        "YearsAtCompany": [years_at_company],
        "OverTime": [overtime],
        "JobSatisfaction": [job_satisfaction],
        "WorkLifeBalance": [work_life_balance],
        "EnvironmentSatisfaction": [environment_satisfaction],
        "YearsSinceLastPromotion": [years_since_last_promotion],
        "PerformanceRating": [performance_rating]
    }
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()

st.subheader("Entered Employee Details")
st.write(input_df)

st.subheader("Prediction Result")

try:
    # Retrieve preprocessor and determine expected feature columns
    if "pre" in model.named_steps:
        preprocessor = model.named_steps["pre"]
        numeric_cols = preprocessor.transformers_[0][2]
        cat_cols = preprocessor.transformers_[1][2]
        expected_features = numeric_cols + cat_cols
    elif hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_.tolist()
    else:
        expected_features = input_df.columns.tolist()

    # Create a full DataFrame with all expected columns
    full_input = pd.DataFrame(columns=expected_features)

    # Fill in known columns from user input, rest with neutral values (0 or "None")
    for col in expected_features:
        if col in input_df.columns:
            full_input[col] = input_df[col]
        else:
            full_input[col] = [0]

    # Reorder columns to match training
    full_input = full_input[expected_features]

    # Predict
    prediction = model.predict(full_input)
    prob = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(full_input)[:, 1][0]

    # Display results
    if prediction[0] == 1:
        if prob is not None:
            st.error(f"🚨 The employee is **LIKELY TO LEAVE**. (Probability: {prob:.2%})")
        else:
            st.error("🚨 The employee is **LIKELY TO LEAVE**.")
    else:
        if prob is not None:
            st.success(f"✅ The employee is **LIKELY TO STAY**. (Probability: {prob:.2%})")
        else:
            st.success("✅ The employee is **LIKELY TO STAY**.")

except Exception as e:
    st.warning(f"⚠️ Could not predict — please ensure your model and input features align.\n\n**Error:** {e}")

st.caption("Developed by Rithish Kumar V | IBM HR Analytics Project 2025")
