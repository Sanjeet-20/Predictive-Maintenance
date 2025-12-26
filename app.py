import streamlit as st
import requests

st.title("Predictive Maintenance Dashboard")

st.write("Enter sensor values to predict machine failure")

# Input fields
sensor14 = st.number_input("Sensor 14", value=8125.55)
sensor9  = st.number_input("Sensor 9", value=9050.17)
sensor4  = st.number_input("Sensor 4", value=1398.21)
sensor7  = st.number_input("Sensor 7", value=553.90)
sensor12 = st.number_input("Sensor 12", value=521.72)

if st.button("Predict Failure"):
    payload = {
        "sensor14": sensor14,
        "sensor9": sensor9,
        "sensor4": sensor4,
        "sensor7": sensor7,
        "sensor12": sensor12
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:
            result = response.json()

            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Failure Probability: {result['failure_probability']}")
        else:
            st.error("API error")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")
