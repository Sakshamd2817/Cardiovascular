import streamlit as st
import joblib
import numpy as np
import pandas as pd # Import pandas for creating DataFrame for scaling


st.set_page_config(page_title="CHD Risk Predictor", layout="centered")

st.title("Cardiovascular Heart Disease (CHD) Risk Predictor")
st.write("This application predicts the 10-year risk of Coronary Heart Disease (CHD) based on various health factors.")

st.markdown("""
---
""")

# Load the trained model and scaler
try:
    model = joblib.load("MODEL.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    st.error("Error: Model file (MODEL.pkl) or scaler file (scaler.pkl) not found. "
             "Please ensure they are in the same directory as this Streamlit app.")
    st.stop()

st.header("Patient Information Input")

# Create input widgets for each feature
# Using columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 30, 80, 50, help="Patient's age in years.")
    male = st.radio("Gender", options=[("Female", 0), ("Male", 1)], format_func=lambda x: x[0], index=1, help="Select Male or Female.")
    currentSmoker = st.radio("Current Smoker", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Is the patient a current smoker?")
    cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=80, value=0, step=1, help="Number of cigarettes smoked per day.")
    BPMeds = st.radio("On Blood Pressure Medication", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Is the patient on blood pressure medication?")
    prevalentStroke = st.radio("Has had a Prevalent Stroke", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Has the patient had a stroke before?")

with col2:
    prevalentHyp = st.radio("Has Prevalent Hypertension", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Does the patient have hypertension?")
    diabetes = st.radio("Has Diabetes", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], help="Does the patient have diabetes?")
    totChol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=10, help="Total cholesterol level.")
    sysBP = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=250, value=120, step=5, help="Systolic blood pressure.")
    diaBP = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=150, value=80, step=5, help="Diastolic blood pressure.")
    BMI = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, format="%.1f", help="Body Mass Index.")
    heartRate = st.number_input("Heart Rate (beats/min)", min_value=40, max_value=150, value=70, step=1, help="Heart rate.")
    glucose = st.number_input("Glucose (mg/dL)", min_value=50, max_value=400, value=100, step=5, help="Glucose level.")

st.markdown("""
---
""")

# Prediction button
if st.button("Predict CHD Risk"):
    # Create a DataFrame for the input, ensuring column order matches training
    # The order of features MUST be the same as X used during training
    input_data = pd.DataFrame([[
        cigsPerDay, BPMeds[1], totChol, BMI, glucose, age, male[1],
        currentSmoker[1], prevalentStroke[1], prevalentHyp[1], diabetes[1],
        sysBP, diaBP, heartRate
    ]], columns=[
        "cigsPerDay","BPMeds","totChol","BMI","glucose","age","male",
        "currentSmoker","prevalentStroke","prevalentHyp","diabetes",
        "sysBP","diaBP","heartRate"
    ])

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[:, 1][0] # Probability of positive class (CHD)

    st.header("Prediction Result")
    if prediction == 1:
        st.error(f"**High Risk of Coronary Heart Disease (CHD)!**")
        st.write(f"Based on the provided information, there is a **{prediction_proba:.2%} probability** of developing CHD within 10 years.")
    else:
        st.success(f"**Low Risk of Coronary Heart Disease (CHD).**")
        st.write(f"Based on the provided information, there is a **{prediction_proba:.2%} probability** of developing CHD within 10 years.")

    st.info("Disclaimer: This prediction is based on a machine learning model and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for any health concerns.")