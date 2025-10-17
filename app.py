import streamlit as st
import joblib
import pandas as pd

# Page setup
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Header
st.title("💼 Employee Salary Prediction")
st.write("Predict employee salary based on input features using a trained ML model.")

# Load model safely
try:
    model = joblib.load("salary_predictor.pkl")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Tabs layout
tab1, tab2, tab3 = st.tabs(["📋 Salary Prediction", "📊 Insights", "ℹ️ About Project"])

# =============== TAB 1 ===============
with tab1:
    st.subheader("📋 Enter Employee Details")

    # 5 features — modify labels only, not logic
    col1, col2 = st.columns(2)
    with col1:
        experience = st.number_input("Experience (in years)", min_value=0, max_value=50, value=2)
        age = st.number_input("Age", min_value=18, max_value=65, value=25)
        education = st.selectbox("Education Level", ["Bachelor", "Master", "PhD"])

    with col2:
        department = st.selectbox("Department", ["HR", "Finance", "Engineering", "Sales", "Marketing"])
        city = st.selectbox("City", ["Bangalore", "Hyderabad", "Chennai", "Mumbai", "Delhi"])

    if st.button("🔍 Predict Salary", use_container_width=True):
        try:
            # Keep input column names same as your model training
            input_data = pd.DataFrame({
                "experience": [experience],
                "age": [age],
                "education": [education],
                "department": [department],
                "city": [city]
            })

            prediction = model.predict(input_data)[0]
            st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =============== TAB 2 ===============
with tab2:
    st.subheader("📊 Insights")
    st.write("""
    This section can later display visual insights such as:
    - Salary distribution across departments  
    - Experience vs Salary correlation  
    - Education level trends  
    """)

# =============== TAB 3 ===============
with tab3:
    st.subheader("ℹ️ About Project")
    st.write("""
    **Project Overview:**  
    This app predicts employee salaries using a trained machine learning model.  
    The model considers 5 key features:
    - Experience  
    - Age  
    - Education Level  
    - Department  
    - City  

    **Tech Stack:**  
    🐍 Python • 📊 Streamlit • 🤖 ML Model (XGBoost/Scikit-learn) • 📦 Joblib  
    """)

st.markdown("---")
st.caption("© 2025 Employee Salary Prediction | Built with Streamlit")
