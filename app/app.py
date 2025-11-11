import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Predictive Maintenance ML", page_icon="⚙️")

model = joblib.load("../model/failure_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

st.title("⚙️ Predictive Maintenance System")
st.write("Enter sensor values to predict if machine maintenance is required.")

temperature = st.number_input("Temperature (°C)", 40, 120)
pressure = st.number_input("Pressure", 30, 70)
humidity = st.number_input("Humidity", 30, 80)
vibration = st.number_input("Vibration Level (0–1)", 0.0, 1.0)
voltage = st.number_input("Voltage (V)", 200, 250)
run_time = st.number_input("Running Hours", 0, 500)

if st.button("Predict Failure"):
    user_input = [[temperature, pressure, humidity, vibration, voltage, run_time]]
    user_scaled = scaler.transform(user_input)
    prediction = model.predict(user_scaled)[0]

    if prediction == 1:
        st.error("🚨 High Failure Risk! Maintenance Required")
    else:
        st.success("✅ Machine is Operating Normally")
