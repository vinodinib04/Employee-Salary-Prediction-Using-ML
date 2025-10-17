import streamlit as st
import pandas as pd
import joblib
from PIL import Image

# ===== 1. PAGE CONFIG =====
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide"
)

# ===== 2. CSS =====
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; font-family: 'Segoe UI', sans-serif; color: #212529; }
.header { text-align:center; font-size:2.5rem; font-weight:700; color:#2c3e50; margin-bottom:0; }
.subheader { text-align:center; font-size:1rem; color:#495057; margin-bottom:2rem; }
.card { background:#fff; padding:2rem; border-radius:1rem; box-shadow:0 4px 12px rgba(0,0,0,0.08); border:1px solid #dee2e6; }
.metric { font-size:1.5rem; font-weight:600; color:#0d6efd; margin-bottom:0.5rem; }
.stButton>button { background-color: #0d6efd; color: white; border: none; border-radius: 0.5rem; padding: 0.7rem 1.5rem; font-size: 1rem; font-weight: 600; }
.stButton>button:hover { background-color: #0b5ed7; }
</style>
""", unsafe_allow_html=True)

# ===== 3. LOAD MODEL =====
@st.cache_data
def load_assets():
    model_data = joblib.load("salary_predictor.pkl")
    eval_plot = Image.open("images/plot.png")
    return model_data, eval_plot

model_data, eval_plot = load_assets()
model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]

# ===== 4. HEADER =====
st.markdown('<p class="header">Salary Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Predict an employee\'s annual salary based on key factors.<br>Use the form to enter details and get instant insights.</p>', unsafe_allow_html=True)

# ===== 5. TWO-PANE DASHBOARD =====
col1, col2 = st.columns([1,1], gap="large")

# ---- LEFT PANE: FORM (3 features for prediction) ----
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Employee Input")
    with st.form("salary_form"):
        # 3 features for prediction
        experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
        education = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
        job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)
        
        submitted = st.form_submit_button("Predict Salary")
    st.markdown('</div>', unsafe_allow_html=True)

# ---- RIGHT PANE: OUTPUT + 2 features displayed ----
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💰 Prediction & Insights")

    if submitted:
        try:
            input_data = {
                "Years of Experience": experience,
                "Education Level": education,
                "Job Title": job_title,
                # For model completeness, fill placeholders if model expects more columns
                "Age": 30,
                "Gender": label_encoders["Gender"].classes_[0]
            }
            input_df = pd.DataFrame([input_data])
            for col in ["Education Level", "Job Title", "Gender"]:
                input_df[col] = label_encoders[col].transform(input_df[col])
            input_scaled = scaler.transform(input_df)
            predicted_salary = model.predict(input_scaled)[0]

            st.markdown(f'<p class="metric">Predicted Salary: ₹{predicted_salary*13:,.0f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric">Experience: {experience} yrs</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="metric">Education: {education}</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.markdown('<p class="metric">Enter details to see prediction</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ===== 6. FOOTER =====
st.markdown("---")
st.caption("© 2025 | Salary Prediction Dashboard | Built with Streamlit 💼")
