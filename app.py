import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 🔹 Load dataset
data = pd.read_csv("student_performance.csv")
data.rename(columns={
    'studytime': 'study_time',
    'famrel': 'family_support',
    'absences': 'absences',
    'G1': 'previous_grade',
    'G3': 'final_grade'
}, inplace=True)
data['family_support'] = data['family_support'].astype(str).str.capitalize()

# 🔹 Features & target (example columns, adjust to your dataset)
X = data[['study_time', 'family_support', 'absences', 'previous_grade']]
y = data['final_grade']

# 🔹 Handle categorical (family_support is categorical)
categorical_features = ['family_support']
numeric_features = ['study_time', 'absences', 'previous_grade']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', StandardScaler(), numeric_features)
])

# 🔹 Choose model: Linear Regression OR Decision Tree
model = DecisionTreeRegressor(random_state=42)
# model = LinearRegression()

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', model)
])

# 🔹 Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Train
pipeline.fit(X_train, y_train)

# 🔹 Evaluate
y_pred = pipeline.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 🔹 Save trained model
with open("student_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as student_model.pkl")


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