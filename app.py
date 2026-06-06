import streamlit as st
import pandas as pd
import joblib

# Load models
risk_model = joblib.load('risk_model.pkl')
salary_model = joblib.load('salary_model.pkl')

st.set_page_config(page_title="AI Job Impact Predictor", layout="wide")

st.title("🌍 AI vs Jobs: Career Forecaster")
st.markdown("Predict automation risk and salary trends based on your skill profile.")

# Sidebar for inputs
st.sidebar.header("User Profile")

job_title = st.sidebar.selectbox("Job Title", ['Data Scientist', 'Software Engineer', 'Data Analyst',
       'DevOps Engineer', 'Cybersecurity Analyst', 'Cloud Engineer',
       'Business Analyst', 'AI Researcher', 'ML Engineer',
       'Product Manager'])
country = st.sidebar.selectbox("Country", ["USA", "India", "Canada", "UK", "Germany", "Australia"])
exp_level = st.sidebar.selectbox("Experience Level", ["Entry", "Mid", "Senior"])
edu_level = st.sidebar.selectbox("Education", ["Bachelor", "Master", "PhD"])
skill = st.sidebar.selectbox("Primary Skill", ['Python', 'Java', 'SQL', 'Docker', 'Security', 'AWS', 'Excel', 'Deep Learning', 'Strategy'])

# Create input dataframe
input_data = pd.DataFrame({
    'job_title': [job_title],
    'country': [country],
    'experience_level': [exp_level],
    'education_level': [edu_level],
    'primary_skill': [skill],
})

# Predictions
risk_pred = risk_model.predict(input_data)[0]
salary_pred = salary_model.predict(input_data)[0]

# Display Results
col1, col2 = st.columns(2)

with col1:
    st.metric("Predicted Automation Risk", f"{risk_pred*100:.1f}%")
    if risk_pred > 0.7:
        st.error("High Risk: Consider upskilling in creative or strategic areas.")
    elif risk_pred > 0.4:
        st.warning("Moderate Risk: AI will likely augment this role.")
    else:
        st.success("Low Risk: This role has high survival probability.")

with col2:
    st.metric("Predicted Monthly Salary", f"${salary_pred:,.2f}")

# Visualization
st.divider()
st.subheader("Strategic Insights")
st.write(f"A {exp_level}-level {job_title} in {country} is projected to earn ${salary_pred*12:,.2f} annually.")