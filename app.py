# ======================================================================================
# 1. IMPORTS
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================================================
# 2. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(page_title="Salary Prediction System", page_icon="💼", layout="centered")

# ======================================================================================
# 3. CUSTOM CSS (Light, Formal Theme)
# ======================================================================================
st.markdown("""
<style>
.stApp { background-color: #f8f9fa; color: #212529; font-family: 'Segoe UI', sans-serif; }
.header-text { text-align: center; font-weight: 700; font-size: 2.2rem; color: #2c3e50; margin-bottom: 0.2rem; }
.subheader-text { text-align: center; color: #495057; font-size: 1rem; margin-bottom: 1.5rem; }
.form-container { background: #ffffff; padding: 2rem; border-radius: 0.8rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 1px solid #dee2e6; }
.stButton>button { background-color: #0d6efd; color: white; border: none; border-radius: 0.4rem; padding: 0.6rem 1.4rem; font-size: 1rem; font-weight: 600; }
.stButton>button:hover { background-color: #0b5ed7; }
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
# 5. SESSION STATE FOR DYNAMIC CHARTS
# ======================================================================================
if "predictions_log" not in st.session_state:
    st.session_state.predictions_log = pd.DataFrame(columns=["Age","Gender","Education Level","Job Title","Years of Experience","Predicted Salary"])

# ======================================================================================
# 6. TABS LAYOUT
# ======================================================================================
tab1, tab2, tab3 = st.tabs(["📋 Salary Prediction", "📊 Insights", "ℹ️ About Project"])

# ======================================================================================
# TAB 1 - SALARY PREDICTION
# ======================================================================================
with tab1:
    st.markdown('<p class="header-text">Salary Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="subheader-text">Predict an employee\'s annual salary using ML</p>', unsafe_allow_html=True)

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

            # Save prediction in session_state for Insights tab
            log_entry = input_data.copy()
            log_entry["Predicted Salary"] = predicted_salary
            st.session_state.predictions_log = pd.concat([st.session_state.predictions_log, pd.DataFrame([log_entry])], ignore_index=True)

            st.success(f"💰 Predicted Annual Salary: ₹{predicted_salary*13:,.0f}")
            st.image(eval_plot, caption="Model Evaluation: Actual vs Predicted Salaries", use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ======================================================================================
# TAB 2 - INSIGHTS (DYNAMIC)
# ======================================================================================
# TAB 2 - INSIGHTS (DYNAMIC)
with tab2:
    st.subheader("📊 Insights")
    df = st.session_state.predictions_log

    if df.empty:
        st.info("Make some predictions first to see dynamic insights here!")
    else:
        # Scatter plot: Experience vs Salary
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x="Years of Experience", y="Predicted Salary", hue="Job Title", s=100, ax=ax1)
        ax1.set_title("Predicted Salary vs Experience")
        st.pyplot(fig1)

        # Bar chart: Average Salary by Job Title
        fig2, ax2 = plt.subplots()
        df.groupby("Job Title")["Predicted Salary"].mean().plot(kind="bar", color="#0d6efd", ax=ax2)
        ax2.set_ylabel("Average Salary")
        ax2.set_title("Average Predicted Salary by Job Title")
        st.pyplot(fig2)

# ======================================================================================
# TAB 3 - ABOUT PROJECT
# ======================================================================================
with tab3:
    st.subheader("ℹ️ About Project")
    st.write("""
    **Project Overview:**  
    This app predicts employee salaries using a trained ML model.

    **Features Used:**
    - Age  
    - Gender  
    - Education Level  
    - Job Title  
    - Years of Experience  

    **Tech Stack:**  
    Python • Streamlit • Scikit-learn • Joblib
    """)

st.markdown("---")
st.caption("© 2025 | Employee Salary Prediction | Built with Streamlit 💼")
