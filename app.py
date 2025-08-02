import streamlit as st
import numpy as np
import pandas as pd
import pickle

# 🔹 Load trained model
model = pickle.load(open("student_model.pkl", "rb"))

st.title("🎓 Student Performance Prediction")

# 🔹 Form Inputs
study_time = st.slider("Study Time (hours per week)", 0, 20)
family_support = st.selectbox("Family Support", ["Yes", "No"])
absences = st.slider("Number of Absences", 0, 50)
previous_grade = st.slider("Previous Grade", 0, 100)

# 🔹 Convert categorical
family_support_val = "Yes" if family_support == "Yes" else "No"

# 🔹 Predict
if st.button("Predict Exam Score"):
    
    input_data = pd.DataFrame([[study_time, family_support_val, absences, previous_grade]], 
                              columns=['study_time', 'family_support', 'absences', 'previous_grade'])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Exam Score: {prediction:.2f}")