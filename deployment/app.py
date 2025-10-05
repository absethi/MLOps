import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import os

# -----------------------------
# Load the trained tourism model
# -----------------------------
model_path = hf_hub_download(
    repo_id="absethi1894/MLOps",                 # 👈 consistent with pipeline
    filename="best_tourism_model_v1.joblib",     # 👈 consistent name
    token=os.getenv("HF_TOKEN")                  # if private repo
)
model = joblib.load(model_path)

# -----------------------------
# Streamlit App UI
# -----------------------------
st.title("Tourism Product Purchase Prediction")
st.write("""
This application predicts the likelihood of a customer purchasing the tourism package
based on their profile and travel-related information.
Please provide the details below:
""")

# -----------------------------
# User Inputs
# -----------------------------
gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
age = st.number_input("Age", min_value=18, max_value=80, value=30)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
monthly_income = st.number_input("Monthly Income (INR)", min_value=10000, max_value=500000, value=50000, step=1000)

num_trips = st.number_input("Number of Trips", min_value=0, max_value=50, value=1)
num_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
num_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)

passport = st.selectbox("Passport", [0, 1])
own_car = st.selectbox("Own Car", [0, 1])

duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=100, value=15)
num_followups = st.number_input("Number of Followups", min_value=0, max_value=10, value=1)
preferred_star = st.selectbox("Preferred Property Star", [3, 4, 5])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
pitch_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

# -----------------------------
# Create DataFrame for prediction
# -----------------------------
input_data = pd.DataFrame([{
    'Gender': gender,
    'MaritalStatus': marital_status,
    'Age': age,
    'Designation': designation,
    'Occupation': occupation,
    'MonthlyIncome': monthly_income,
    'NumberOfTrips': num_trips,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfChildrenVisiting': num_children,
    'Passport': passport,
    'OwnCar': own_car,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_star,
    'ProductPitched': product_pitched,
    'PitchSatisfactionScore': pitch_score
}])

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
