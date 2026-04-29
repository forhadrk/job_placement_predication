import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests

# Set the browser title and favicon
st.set_page_config(
    page_title="Heart Disease Predictor",  # Browser tab title
    page_icon="❤️",  # Favicon (emoji or path to an image file)
    layout="centered",  # Layout can be "centered" or "wide"
)
# Load the saved preprocessor and model
#preprocessor, model = joblib.load('pipeline_and_model.pkl')
model_path = "pipeline_and_model.pkl"
model_url = "https://your-storage-service.com/pipeline_and_model.pkl"

if not os.path.exists(model_path):
    response = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(response.content)

preprocessor, model = joblib.load(model_path)

# Define the column names for input data (match training data)
input_columns = [
    'age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak', 
    'gender', 'fastingbloodsugar', 'exerciseangia', 
    'chestpain', 'restingrelectro', 'slope', 'noofmajorvessels'
]

# Dictionary to map numerical values to descriptive labels
chestpain_options = {
    0: "Low",
    1: "Normal",
    2: "Medium",
    3: "High"
}

# Title and description
st.title("Heart Disease Prediction App")
st.write("Enter the following information to predict the presence of heart disease.")

# Layout for input fields: Two inputs per row
col1, col2 = st.columns(2)

# Age and Resting Blood Pressure
with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
with col2:
    restingBP = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)

# Serum Cholesterol and Maximum Heart Rate
with col1:
    serumcholestrol = st.number_input("Serum Cholestrol (mg/dl)", min_value=100, max_value=600, value=200)
with col2:
    maxheartrate = st.number_input("Maximum Heart Rate", min_value=50, max_value=220, value=150)

# Oldpeak and Gender
with col1:
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0)
with col2:
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")

# Fasting Blood Sugar and Exercise-Induced Angina
with col1:
    fastingbloodsugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
with col2:
    exerciseangia = st.selectbox("Exercise-Induced Angina", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Chest Pain Type and Resting Electrocardiographic Results
with col1:
    chestpain =  st.selectbox(
    "Chest Pain Type",
    options=list(chestpain_options.keys()),  # Use numerical keys as options
    format_func=lambda x: f"{x} - {chestpain_options[x]}"  # Format for display
)
    #st.selectbox("Chest Pain Type", [0, 1, 2, 3])
with col2:
    restingrelectro_options = {
        0: "Normal",
        1: "Having ST-T Wave Abnormality",
        2: "Other Abnormalities"
    }
    restingrelectro = st.selectbox(
        "Resting Electrocardiographic Results",
        options=list(restingrelectro_options.keys()),
        format_func=lambda x: f"{x} - {restingrelectro_options[x]}"
    )

# Slope of the Peak Exercise ST Segment and Number of Major Vessels
with col1:
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
with col2:
    noofmajorvessels = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

# Prediction button
if st.button("Predict"):
    # Prepare input data as a DataFrame with correct column names
    input_data = pd.DataFrame([[
        age, restingBP, serumcholestrol, maxheartrate, oldpeak, 
        gender, fastingbloodsugar, exerciseangia, 
        chestpain, restingrelectro, slope, noofmajorvessels
    ]], columns=input_columns)

    # Preprocess the input data
    try:
        processed_input = preprocessor.transform(input_data)

        # Predict using the model
        prediction = model.predict(processed_input)

        # Display the result
        # result = "Presence of Heart Disease" if prediction[0] == 1 else "Absence of Heart Disease"
        # st.subheader(f"Prediction: {result}")
        if prediction[0] == 1:
            st.markdown(
                """<h3 style='color:red;text-align:center;'>Prediction: Presence of Heart Disease</h3>""",
                unsafe_allow_html=True
            )
        else:
            st.subheader("Prediction: Absence of Heart Disease")
    except ValueError as e:
        st.error(f"Error during prediction: {e}")

# Footer or additional notes
#st.write("This app uses an XGBoost model trained on heart disease data to provide predictions. Ensure accurate input for best results.")
#st.title("Predict Heart Disease Risk")
#st.subheader("An AI-powered app to predict the presence of heart disease.")
#st.write("This app uses a machine learning model trained on heart disease data to provide predictions.")
