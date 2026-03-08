import streamlit as st
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

st.title("🎓 Student Performance Predictor")

# ---------------- LOAD PICKLE FILES ----------------

difficulty_encoder = pickle.load(open("models/Difficulty_Level_label_encoder.pkl", "rb"))
parent_encoder = pickle.load(open("models/Parent_Education_Level_label_encoder.pkl", "rb"))
income_encoder = pickle.load(open("models/Family_Income_Level_label_encoder.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

# ---------------- USER INPUT ----------------

difficulty = st.selectbox("Difficulty Level", ["Easy", "Moderate", "Hard"])

attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)

midterm = st.slider("Midterm Score", 0.0, 100.0, 50.0)

assignments = st.slider("Assignments Average", 0.0, 100.0, 60.0)

quizzes = st.slider("Quizzes Average", 0.0, 100.0, 60.0)

participation = st.slider("Participation Score", 0.0, 10.0, 5.0)

projects = st.slider("Projects Score", 0.0, 30.0, 15.0)

study_hours = st.slider("Study Hours per Week", 0.0, 40.0, 10.0)

parent_edu = st.selectbox(
    "Parent Education Level",
    ["Primary", "Secondary", "Graduate", "Postgraduate"]
)

income = st.selectbox(
    "Family Income Level",
    ["Low", "Mid", "High"]
)

stress = st.slider("Stress Level", 0.0, 10.0, 5.0)

sleep = st.slider("Sleep Hours per Night", 0.0, 12.0, 7.0)

branch = st.selectbox("Branch", ["CSE", "Civil", "ECE", "EEE", "ME"])

internet = st.selectbox("Internet Access at Home", ["Yes", "No"])

# ---------------- ENCODING ----------------

difficulty_encoded = difficulty_encoder.transform([difficulty])[0]
parent_encoded = parent_encoder.transform([parent_edu])[0]
income_encoded = income_encoder.transform([income])[0]

internet_encoded = 1 if internet == "Yes" else 0

# ---------------- ONE HOT BRANCH ----------------

Branch_CSE = 1 if branch == "CSE" else 0
Branch_Civil = 1 if branch == "Civil" else 0
Branch_ECE = 1 if branch == "ECE" else 0
Branch_EEE = 1 if branch == "EEE" else 0
Branch_ME = 1 if branch == "ME" else 0

# ---------------- BUTTON ----------------

if st.button("Predict Final Score"):

    input_data = pd.DataFrame([{
        "Difficulty_Level": difficulty_encoded,
        "Attendance (%)": attendance,
        "Midterm_Score": midterm,
        "Assignments_Avg": assignments,
        "Quizzes_Avg": quizzes,
        "Participation_Score": participation,
        "Projects_Score": projects,
        "Study_Hours_per_Week": study_hours,
        "Parent_Education_Level": parent_encoded,
        "Family_Income_Level": income_encoded,
        "Stress_Level": stress,
        "Sleep_Hours_per_Night": sleep,
        "Branch_CSE": Branch_CSE,
        "Branch_Civil": Branch_Civil,
        "Branch_ECE": Branch_ECE,
        "Branch_EEE": Branch_EEE,
        "Branch_ME": Branch_ME,
        "Internet_Access_at_Home_Yes": internet_encoded
    }])

    # Remove Branch_CSE because model wasn't trained with it
    input_data = input_data[
        ['Difficulty_Level', 'Attendance (%)', 'Midterm_Score', 'Assignments_Avg',
         'Quizzes_Avg', 'Participation_Score', 'Projects_Score',
         'Study_Hours_per_Week', 'Parent_Education_Level', 'Family_Income_Level',
         'Stress_Level', 'Sleep_Hours_per_Night',
         'Branch_Civil', 'Branch_ECE', 'Branch_EEE', 'Branch_ME',
         'Internet_Access_at_Home_Yes']
    ]

    st.subheader("Processed Input Data")
    st.dataframe(input_data)

    # ---------------- MODEL PREDICTION ----------------

    prediction = model.predict(input_data)

    st.success(f"Predicted Final Score: {prediction[0]:.2f}")