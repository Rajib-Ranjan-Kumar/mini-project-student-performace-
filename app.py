import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import os
import joblib
import requests



st.title("🎓 Student Performance Predictor + AI Advisor")

# ---------------- HUGGING FACE API ----------------

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# ---------------- LOAD MODEL ----------------

MODEL_PATH = os.path.join("models", "model.pkl")
ENC1 = os.path.join("models", "Difficulty_Level_label_encoder.pkl")
ENC2 = os.path.join("models", "Parent_Education_Level_label_encoder.pkl")
ENC3 = os.path.join("models", "Family_Income_Level_label_encoder.pkl")

difficulty_encoder = joblib.load(ENC1)
parent_encoder = joblib.load(ENC2)
income_encoder = joblib.load(ENC3)
model = joblib.load(MODEL_PATH)

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

Branch_CSE = 1 if branch == "CSE" else 0
Branch_Civil = 1 if branch == "Civil" else 0
Branch_ECE = 1 if branch == "ECE" else 0
Branch_EEE = 1 if branch == "EEE" else 0
Branch_ME = 1 if branch == "ME" else 0


# ---------------- AI RECOMMENDATION FUNCTION ----------------

def generate_recommendation(student_data, predicted_score):

    prompt = f"""
    Student Data: {student_data}
    Predicted Final Score: {predicted_score}

    Give helpful recommendations to improve the student's academic performance.
    Include study strategies, stress management, and time management tips.
    """

    output = query({
        "inputs": prompt,
        "parameters": {"max_length": 200}
    })

    try:
        return output[0]["generated_text"]
    except:
        return "AI recommendation currently unavailable."


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

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Final Score: {prediction:.2f}")

    with st.spinner("Generating AI recommendations..."):
        recommendation = generate_recommendation(input_data.to_dict(), prediction)

    st.subheader("📚 AI Study Recommendations")
    st.write(recommendation)