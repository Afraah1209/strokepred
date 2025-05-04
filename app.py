
import streamlit as st
import pandas as pd
import joblib

st.title("Stroke Prediction App")

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(data):
    df = pd.DataFrame([data])
    df = df.astype(float)
    df_scaled = scaler.transform(df)
    result = model.predict(df_scaled)
    return result[0]

input_data = {
    "gender": st.selectbox("Gender", [0, 1, 2]),
    "age": st.slider("Age", 0, 100),
    "hypertension": st.selectbox("Hypertension", [0, 1]),
    "heart_disease": st.selectbox("Heart Disease", [0, 1]),
    "ever_married": st.selectbox("Ever Married", [0, 1]),
    "work_type": st.selectbox("Work Type", [0, 1, 2, 3, 4]),
    "Residence_type": st.selectbox("Residence Type", [0, 1]),
    "avg_glucose_level": st.number_input("Average Glucose Level"),
    "bmi": st.number_input("BMI"),
    "smoking_status": st.selectbox("Smoking Status", [0, 1, 2, 3])
}

if st.button("Predict Stroke"):
    result = predict(input_data)
    st.success("Stroke Risk: Yes" if result == 1 else "Stroke Risk: No")
