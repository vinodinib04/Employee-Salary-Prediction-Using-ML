# ======================================================================================
# 1. IMPORTS
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import json
from streamlit_lottie import st_lottie


# ======================================================================================
# 2. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(
    page_title="Salary Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================================================
# 3. CUSTOM STYLING (Neon + Glassmorphism)
# ======================================================================================
st.markdown("""
<style>
    /* Background & base color theme */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e0e7ff;
        font-family: 'Poppins', sans-serif;
    }

    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 1rem;
        padding: 2rem;
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 25px rgba(124, 58, 237, 0.2);
    }

    /* Headings */
    .header-text {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subheader-text {
        text-align: center;
        color: #c7d2fe;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Input fields */
    input, select, textarea {
        background-color: rgba(255,255,255,0.1) !important;
        color: #e0e7ff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 0.5rem !important;
    }

    /* Button */
    .stButton>button {
        background-image: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        font-weight: 600;
        border-radius: 0.6rem;
        padding: 0.7rem 1.2rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px #818cf8;
    }

    /* Metrics */
    .stMetric > label { color: #a5b4fc; }
    .stMetric > div > span { color: #c084fc; font-size: 2rem; font-weight: 700; }

    /* Expander */
    .stExpander {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #94a3b8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# 4. LOAD ASSETS
# ======================================================================================
@st.cache_data
def load_all_assets():
    model_data = joblib.load("salary_predictor.pkl")
    eval_plot = Image.open("images/plot.png")
    with open("animation.json", "r") as f:
        lottie_json = json.load(f)
    return model_data, eval_plot, lottie_json

model_data, eval_plot, lottie_json = load_all_assets()
model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]


# ======================================================================================
# 5. SESSION STATE
# ======================================================================================
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.predicted_salary = 0.0


# ======================================================================================
# 6. HEADER
# ======================================================================================
st.markdown('<p class="header-text">🔮 AI Salary Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Predict employee salaries using Machine Learning magic ✨</p>', unsafe_allow_html=True)
st.divider()


# ======================================================================================
# 7. MAIN LAYOUT
# ======================================================================================
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    with st.form("salary_form"):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;color:#c7d2fe;'>Employee Profile</h3>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_, index=2)
        with c2:
            years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
            gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
        
        job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_, index=5)
        submit_button = st.form_submit_button("✨ Predict Salary")
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if not st.session_state.prediction_made:
        st_lottie(lottie_json, height=280, key="ai_oracle")
        st.info("Fill in the details and click Predict to unveil your salary range 👇")
    else:
        salary = st.session_state.predicted_salary
        st.metric("Estimated Annual Salary Range", f"${salary * 0.925:,.0f} - ₹{salary * 1.075:,.0f}", "Based on your profile")
        st.success("Prediction successful 🎯")
    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================================
# 8. PREDICTION LOGIC
# ======================================================================================
if submit_button:
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
    st.session_state.predicted_salary = predicted_salary
    st.session_state.prediction_made = True
    st.rerun()


# ======================================================================================
# 9. FOOTER
# ======================================================================================
st.markdown("---")
with st.expander("View Model Performance 📊"):
    st.image(eval_plot, caption="Model Evaluation: Actual vs. Predicted Salaries", use_container_width=True)
    st.info("This plot compares predicted vs actual salaries — the closer to the diagonal line, the better the accuracy.")





