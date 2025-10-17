# ======================================================================================
# 1. IMPORTS
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib

# ======================================================================================
# 2. PAGE CONFIGURATION
# ======================================================================================
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered"
)

# ======================================================================================
# 3. LOAD MODEL AND ASSETS
# ======================================================================================
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load("salarypredict.pkl")  # model pickle contains model, label_encoders, scaler
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_dict = load_model()
if model_dict:
    model = model_dict["model"]
    label_encoders = model_dict["label_encoders"]
    scaler = model_dict.get("scaler", None)
else:
    model = None
    label_encoders = None
    scaler = None

# ======================================================================================
# 4. APP HEADER
# ======================================================================================
st.title("💼 Employee Salary Prediction")
st.markdown("""
Welcome to the **Employee Salary Prediction App**!  
Predict an employee's salary based on their experience, age, education, gender, and department.
""")

# ======================================================================================
# 5. INPUT FORM
# ======================================================================================
st.subheader("Enter Employee Details")
col1, col2 = st.columns(2)

with col1:
    experience = st.number_input("Experience (in years)", min_value=0, max_value=50, value=3)
    # Use only encoder classes to prevent unseen label error
    education = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)

with col2:
    age = st.number_input("Age", min_value=18, max_value=70, value=25)
    gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
    department = st.selectbox("Department", options=label_encoders["Job Title"].classes_)

# ======================================================================================
# 6. PREDICTION
# ======================================================================================
st.markdown("---")
if st.button("🔍 Predict Salary"):
    if model:
        # Build input DataFrame with original training column names
        input_df = pd.DataFrame({
            "Age": [age],
            "Gender": [gender],
            "Education Level": [education],
            "Job Title": [department],
            "Years of Experience": [experience]
        })

        # Transform categorical columns
        for col in ["Gender", "Education Level", "Job Title"]:
            input_df[col] = label_encoders[col].transform(input_df[col])

        # Scale numeric columns if scaler exists
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values

        # Predict
        try:
            prediction = model.predict(input_scaled)
            st.success(f"💰 **Predicted Salary:** ₹ {prediction[0]:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error("Model not loaded. Make sure 'salarypredict.pkl' exists.")

# ======================================================================================
# 7. FOOTER
# ======================================================================================
st.markdown("---")
st.caption("Developed by **Vinodini Bandaru** | Powered by Machine Learning 🚀")
