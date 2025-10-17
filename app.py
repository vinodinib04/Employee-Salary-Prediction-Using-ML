import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_smote.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# -------------------------------
# App Title & Description
# -------------------------------
st.title("💼 Employee Salary Prediction")
st.markdown("""
Welcome to the **Employee Salary Prediction App**!  
This tool predicts an employee's salary based on their experience, age, education, and department.

Please fill in the details below and click **Predict Salary** to see the estimated amount.
""")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Experience (in years)", min_value=0, max_value=50, value=3)
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

with col2:
    age = st.number_input("Age", min_value=18, max_value=70, value=25)
    department = st.selectbox("Department", ["Sales", "Engineering", "HR", "Marketing", "Finance"])

# -------------------------------
# Prediction
# -------------------------------
st.markdown("---")
if st.button("🔍 Predict Salary"):
    if model is not None:
        input_data = pd.DataFrame({
            "experience": [experience],
            "age": [age],
            "education": [education],
            "department": [department]
        })

        try:
            prediction = model.predict(input_data)
            st.success(f"💰 **Predicted Salary:** ₹ {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not found. Please make sure 'xgboost_smote.pkl' is in the project folder.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by **Vinodini Bandaru** | Powered by Machine Learning 🚀")
