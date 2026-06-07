import streamlit as st
import pandas as pd
import joblib

df = pd.read_csv('Future of Jobs AI Dataset.csv')
risk_model = joblib.load('risk_model.pkl')
salary_model = joblib.load('salary_model.pkl')

st.set_page_config(page_title="AI Job Impact Predictor", layout="wide")
st.title("🌍 AI vs Jobs: Skill Profile Forecaster")

# Sidebar
st.sidebar.header("User Profile")
job_title = st.sidebar.selectbox("Job Title", sorted(df['job_title'].unique()))
country = st.sidebar.selectbox("Country", sorted(df['country'].unique()))
exp_level = st.sidebar.selectbox("Experience Level", ['Entry', 'Mid', 'Senior'])
edu_level = st.sidebar.selectbox("Education", ['Bachelor', 'Master', 'PhD'])
skill = st.sidebar.selectbox("Primary Skill", sorted(df['primary_skill'].unique()))

# Input Mapping
input_data = pd.DataFrame({
    'job_title': [job_title],
    'country': [country],
    'experience_level': [exp_level],
    'education_level': [edu_level],
    'primary_skill': [skill]
})

risk_pred = risk_model.predict(input_data)[0]
salary_pred = salary_model.predict(input_data)[0]

# Metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Predicted Automation Risk", f"{risk_pred*100:.1f}%")
    if risk_pred > 0.7:
        st.error("High Risk Scenario: Targeted upskilling recommended.")
    elif risk_pred > 0.4:
        st.warning("Moderate Risk Scenario: Structural augmentation expected.")
    else:
        st.success("Low Risk Scenario: Strong structural stability.")

with col2:
    st.metric("Predicted Monthly Salary", f"${salary_pred:,.2f}")

st.divider()

# Section 1: Insights by Experience Level
st.subheader(f"📊 Insights for {job_title}")

filtered_df = df[df['job_title'] == job_title]

col3, col4 = st.columns(2)
with col3:
    st.markdown("**Average Automation Risk by Experience Level**")
    risk_by_exp = filtered_df.groupby('experience_level')['ai_risk_score'].mean() * 100
    risk_by_exp = risk_by_exp.reindex(['Entry', 'Mid', 'Senior'])
    st.bar_chart(risk_by_exp)

with col4:
    st.markdown("**Average Monthly Salary by Experience Level**")
    salary_by_exp = filtered_df.groupby('experience_level')['salary'].mean()
    salary_by_exp = salary_by_exp.reindex(['Entry', 'Mid', 'Senior'])
    st.bar_chart(salary_by_exp)

st.divider()
# Grouping and ordering the metrics natively
edu_grouped = df.groupby('education_level')[['ai_risk_score', 'salary']].mean()
edu_grouped = edu_grouped.reindex(['Bachelor', 'Master', 'PhD'])

col5, col6 = st.columns(2)
with col5:
    st.markdown("**Global AI Automation Risk by Degree Level (%)**")
    st.bar_chart(edu_grouped['ai_risk_score'] * 100)

with col6:
    st.markdown("**Global Average Monthly Salary by Degree Level ($)**")
    st.bar_chart(edu_grouped['salary'])