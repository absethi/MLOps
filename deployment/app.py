import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import os

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "tourism_xgb_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please check deployment.")
else:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Tourism Package Purchase Prediction 🧳")

st.write("Fill in the details below to predict whether a customer will purchase a package.")

# Example input fields (adjust based on your dataset features)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration = st.number_input("Duration of Stay (days)", min_value=1, max_value=30, value=5)
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=50000)
num_person_visiting = st.slider("Number of People Visiting", min_value=1, max_value=10, value=2)
designations = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
gender = st.radio("Gender", ["Male", "Female"])

# -----------------------------
# Prepare input for model
# -----------------------------
input_dict = {
    "Age": age,
    "DurationOfPitch": duration,
    "MonthlyIncome": monthly_income,
    "NumberOfPersonVisiting": num_person_visiting,
    "Designation": designations,
    "Gender": gender,
}

input_data = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        result = "Will Purchase Package ✅"
    else:
        result = "Will Not Purchase Package ❌"
    
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
