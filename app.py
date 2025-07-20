# ======================================================================================
# 1. IMPORTS
# All necessary libraries are imported at the top.
# ======================================================================================
import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import json
from streamlit_lottie import st_lottie


# ======================================================================================
# 2. PAGE CONFIGURATION
# Sets the browser tab title, icon, and layout.
# ======================================================================================
st.set_page_config(
    page_title="Salary AI - Showcase Version",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================================================
# 3. CUSTOM CSS
# This block injects custom CSS for advanced styling.
# ======================================================================================
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: url("https://www.transparenttextures.com/patterns/dark-matter.png");
        background-color: #0c111c;
    }
    .main .block-container { padding: 2rem 3rem; }
    .header-text { font-family: 'Garamond', serif; color: #e0e7ff; font-weight: 700; font-size: 3rem; text-align: center; }
    .subheader-text { color: #a5b4fc; font-size: 1.1rem; text-align: center; margin-bottom: 2.5rem; }
    .form-container { background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(10px); padding: 2rem; border-radius: 1rem; border: 1px solid rgba(255, 255, 255, 0.1); }
    .stButton>button { background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 0.8rem 1.6rem; border-radius: 0.75rem; font-weight: bold; font-size: 1.1rem; border: none; }
    .stMetric { background-color: transparent; border-radius: 1rem; padding: 1rem; text-align: center; }
    .stMetric > label { font-weight: 600; color: #c7d2fe; }
    .stMetric > div > span { font-size: 2.5rem; color: #818cf8; font-weight: 700; }
    .stExpander { background: rgba(255, 255, 255, 0.05); border-radius: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.1); }
    .footer { text-align: center; padding: 2rem; color: #9ca3af; }
    .footer a { color: #818cf8; text-decoration: none; font-weight: 600; margin: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ======================================================================================
# 4. ASSET LOADING
# This function loads all assets and is cached for performance.
# This is the most critical part for preventing errors.
# ======================================================================================
@st.cache_data
def load_all_assets():
    """Loads all necessary assets from local files."""
    # Load the trained model and associated objects
    model_data = joblib.load("salary_predictor.pkl")
    
    # Load the evaluation plot image
    eval_plot = Image.open("images/plot.png")
    
    # Load the Lottie animation from a local JSON file
    with open("animation.json", "r") as f:
        lottie_json = json.load(f)
        
    return model_data, eval_plot, lottie_json

# Call the function RIGHT AWAY to load everything into variables.
# This solves the NameError because these variables will exist before the UI is built.
model_data, eval_plot, lottie_json = load_all_assets()

# Unpack the model data dictionary into individual variables for easy access
model = model_data["model"]
label_encoders = model_data["label_encoders"]
scaler = model_data["scaler"]


# ======================================================================================
# 5. SESSION STATE INITIALIZATION
# Used to store information across reruns, like whether a prediction has been made.
# ======================================================================================
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
    st.session_state.predicted_salary = 0.0


# ======================================================================================
# 6. UI LAYOUT
# ======================================================================================
# Main header
st.markdown('<p class="header-text">🔮 AI Salary Oracle</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Unveil salary predictions with machine learning. Enter the details to begin.</p>', unsafe_allow_html=True)

# Main layout with two columns
col1, col2 = st.columns([1.2, 1], gap="large")

# --- COLUMN 1: INPUT FORM ---
with col1:
    with st.form("salary_form"):
        # Use a markdown div for the styled container
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #c7d2fe;'>Employee Profile</h3>", unsafe_allow_html=True)
        
        # Nested columns for a compact form layout
        form_col1, form_col2 = st.columns(2)
        with form_col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_, index=2)
        with form_col2:
            years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
            gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
        
        job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_, index=5)
        
        # The submit button for the form
        submit_button = st.form_submit_button("✨ Divine the Salary")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- COLUMN 2: OUTPUT AREA (DYNAMIC) ---
with col2:
    # If no prediction has been made yet, show the animation
    if not st.session_state.prediction_made:
        st_lottie(lottie_json, speed=1, height=300, key="initial_animation")
        st.info("Your prediction will appear here once submitted.", icon="💡")
    # If a prediction has been made, show the result
    else:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center; color: #c7d2fe;'>Oracle's Vision</h3>", unsafe_allow_html=True)
        salary = st.session_state.predicted_salary
        
        st.metric(
            label="Estimated Annual Salary Range", 
            value=f"${salary * 0.925:,.0f} - ${salary * 1.075:,.0f}",
            delta="Based on your inputs",
            delta_color="off"
        )
        st.success("The vision is clear! Prediction successful.", icon="✅")
        st.markdown('</div>', unsafe_allow_html=True)


# ======================================================================================
# 7. PREDICTION LOGIC
# This block runs only when the form's submit button is clicked.
# ======================================================================================
if submit_button:
    # Prepare the input data from the form for the model
    input_data = {
        "Age": age, "Gender": gender, "Education Level": education_level,
        "Job Title": job_title, "Years of Experience": years_of_experience
    }
    input_df = pd.DataFrame([input_data])
    
    # Transform categorical data using the loaded label encoders
    for col in ["Gender", "Education Level", "Job Title"]:
        input_df[col] = label_encoders[col].transform(input_df[col])

    # Scale the numerical data using the loaded scaler
    input_scaled = scaler.transform(input_df)
    
    # Make the prediction
    predicted_salary = model.predict(input_scaled)[0]
    
    # Store the result in session state and set the flag
    st.session_state.predicted_salary = predicted_salary
    st.session_state.prediction_made = True
    
    # Fun animation and a page rerun to update the UI
    st.balloons()
    st.rerun()


# ======================================================================================
# 8. FOOTER AND ADDITIONAL INFORMATION
# ======================================================================================
st.markdown("---")
with st.expander(" peek behind the curtain at the model's performance..."):
    st.image(eval_plot, caption="Model Evaluation: Actual vs. Predicted Salaries", use_container_width=True)
    st.info("This plot shows the relationship between the model's predicted salaries and the actual salaries from the test dataset. A strong positive correlation indicates high accuracy.")

st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Crafted with 🧠 & ❤️ by <b>Ayush Anand</b></p>
    <a href="https://github.com/Ayush03A" target="_blank">GitHub</a> | 
    <a href="https://www.linkedin.com/in/3ayushanand/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)

