# ======================================================================================
# 1. IMPORTS
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ======================================================================================
# 2. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(
    page_title="Salary Prediction System",
    page_icon="💼",
    layout="centered"
)

# ======================================================================================
# 3. CUSTOM CSS (Light, Formal Theme)
# ======================================================================================
st.markdown("""
<style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', sans-serif;
    }

    .header-text {
        text-align: center;
        font-weight: 700;
        font-size: 2.2rem;
        color: #2c3e50;
        margin-bottom: 0.2rem;
    }

    .subheader-text {
        text-align: center;
        color: #495057;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .form-container {
        background: #ffffff;
        padding: 2rem;
        border-radius: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #dee2e6;
    }

    input, select, textarea {
        background-color: #ffffff !important;
        color: #212529 !important;
        border: 1px solid #ced4da !important;
        border-radius: 0.4rem !important;
    }

    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border: none;
        border-radius: 0.4rem;
        padding: 0.6rem 1.4rem;
        font-size: 1rem;
        font-weight: 600;
        transition: background 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }

    .footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ======================================================================================
# 4. LOAD ASSETS
# ======================================================================================
@st.cache_data
def load_assets():
    model_data = joblib.load("salary_predictor.pkl")
    eval_plot = Image.open("images/plot.png")
    return model_data, eval_plot

model_data, eval_plot = load_assets()
model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]

# ======================================================================================
# 5. UI HEADER
# ======================================================================================
st.markdown('<p class="header-text">💼 Salary Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Predict an employee\'s annual salary using machine learning</p>', unsafe_allow_html=True)

# ======================================================================================
# 6. INPUT FORM
# ======================================================================================
st.markdown('<div class="form-container">', unsafe_allow_html=True)
with st.form("salary_form"):
    st.header("Employee Details")

    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
    education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
    job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

    submitted = st.form_submit_button("Predict Salary")

st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================
# 7. PREDICTION
# ======================================================================================
if submitted:
    input_data = {
        "Age": age,
        "Gender": gender,
        "Education Level": education_level,
        "Job Title": job_title,
        "Years of Experience": years_of_experience
    }
    input_df = pd.DataFrame([input_data])

    for col in ["Gender", "Education Level", "Job Title"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    predicted_salary = model.predict(input_scaled)[0]

    st.success(f"✅ **Predicted Annual Salary:** ₹{predicted_salary*12:,.0f}")

    st.markdown("---")
    st.image(eval_plot, caption="Model Evaluation: Actual vs. Predicted Salaries", use_container_width=True)








