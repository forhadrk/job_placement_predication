import streamlit as st 
import numpy as np 
import pandas as pd 
import joblib 
import os 
import requests

# =========================
# LOAD MODEL
# =========================
# model = joblib.load("placement_model.pkl")

model_path = "placement_model.pkl"
model_url = "https://your-storage-service.com/placement_model.pkl"

if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

model = joblib.load(model_path)
# =========================
# UI CONFIG
# =========================
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction App")
st.write("Enter student details to predict placement outcome.")

# =========================
# INPUTS (UI IMPROVED)
# =========================
col1, col2 = st.columns(2)

with col1:
    ssc = st.number_input("SSC Percentage", 0.0, 100.0, 50.0)
    degree = st.number_input("Degree Percentage", 0.0, 100.0, 60.0)
    mba = st.number_input("MBA Percentage", 0.0, 100.0, 65.0)
    internship = st.selectbox("Internship Completed", [0, 1], format_func=lambda x: "Yes" if x else "No")
    degree_others = st.selectbox("Undergrad Degree Others", [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    hsc = st.number_input("HSC Percentage", 0.0, 100.0, 55.0)
    emp_test = st.number_input("Employability Test %", 0.0, 100.0, 60.0)
    gender_m = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x else "Female")
    work_exp = st.selectbox("Work Experience", [0, 1], format_func=lambda x: "Yes" if x else "No")
    specialisation = st.selectbox("Mkt & HR Specialisation", [0, 1], format_func=lambda x: "Yes" if x else "No")

# =========================
# PREDICTION
# =========================
if st.button("Predict Placement"):

    input_data = np.array([[
        ssc, hsc, degree,
        emp_test, mba,
        internship, gender_m,
        degree_others, work_exp,
        specialisation
    ]])

    # Regression output
    output = model.predict(input_data)[0]

    st.subheader("📊 Result")

    st.write(f"Raw Score: **{output:.2f}**")

    # Convert regression output into class
    if output >= 0.5:
        st.success("🎉 Student is LIKELY PLACED")
    else:
        st.error("❌ Student is NOT LIKELY placed")
