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
# 3. CUSTOM CSS (Light, Professional Dashboard)
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
    font-size: 2.4rem;
    color: #2c3e50;
    margin-bottom: 0.2rem;
}

.subheader-text {
    text-align: center;
    color: #495057;
    font-size: 1rem;
    margin-bottom: 2rem;
}

.form-container, .prediction-container, .insights-container {
    background: #ffffff;
    padding: 2rem;
    border-radius: 1rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #dee2e6;
    margin-bottom: 2rem;
}

.stButton>button {
    background-color: #0d6efd;
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.7rem 1.5rem;
    font-size: 1rem;
    font-weight: 600;
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
# 4. LOAD MODEL AND ASSETS
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
# 5. HEADER
# ======================================================================================
st.markdown('<p class="header-text">Salary Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Predict an employee\'s annual salary using ML</p>', unsafe_allow_html=True)

# ======================================================================================
# 6. INPUT FORM (Employee Details)
# ======================================================================================
st.markdown('<div class="form-container">', unsafe_allow_html=True)
with st.form("salary_form"):
    st.subheader("Employee Details")
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
    education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
    job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
    submitted = st.form_submit_button("Predict Salary")
st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================
# 7. PREDICTION OUTPUT
# ======================================================================================
if submitted:
    try:
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

        st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
        st.subheader("💰 Predicted Salary")
        st.success(f"₹{predicted_salary*13:,.0f} per year")
        st.image(eval_plot, caption="Model Evaluation: Actual vs Predicted Salaries", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ======================================================================================
# 8. INSIGHTS / NOTES (Professional Dashboard)
# ======================================================================================
st.markdown('<div class="insights-container">', unsafe_allow_html=True)
st.subheader("📌 Insights")
st.markdown("""
- The model uses **Age, Gender, Education Level, Job Title, and Experience** to predict salary.
- Predictions provide an **approximate salary range**; actual salary may vary due to other factors.
""")
st.markdown('</div>', unsafe_allow_html=True)

# ======================================================================================
# 9. FOOTER
# ======================================================================================
st.markdown("---")
st.caption("© 2025 | Employee Salary Prediction | Built with Streamlit 💼")
