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
        model_dict = joblib.load("salary_predictor.pkl")  # Load the pickle file
        return model_dict
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_dict = load_model()

# Extract the actual model from the dictionary
if model_dict is not None:
    model = model_dict.get("model", None)
else:
    model = None

# -------------------------------
# App Title & Description
# -------------------------------
st.title("💼 Employee Salary Prediction")
st.markdown("""
Welcome to the **Employee Salary Prediction App**!  
This tool predicts an employee's salary based on their experience, age, education, gender, and department.

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
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with col2:
    age = st.number_input("Age", min_value=18, max_value=70, value=25)
    department = st.selectbox("Department", ["Sales", "Engineering", "HR", "Marketing", "Finance"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Salary"):
    if model_dict is not None:
        # Build input DataFrame with original column names
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [department],  # Department corresponds to Job Title
            "Years of Experience": [experience]
        })

        # Transform categorical columns using label encoders from the model dict
        label_encoders = model_dict.get("label_encoders", {})
        for col in ["Gender", "Education Level", "Job Title"]:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numeric features if scaler exists
        scaler = model_dict.get("scaler", None)
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        # Make prediction
        try:
            prediction = model.predict(input_scaled)
            st.success(f"💰 **Predicted Salary:** ₹ {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not found. Please make sure 'salarypredict.pkl' is in the project folder.")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Developed by **Vinodini Bandaru** | Powered by Machine Learning 🚀")
