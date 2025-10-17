# ======================================================================================
# 6. MAIN TABS
# ======================================================================================
tab1, tab2, tab3 = st.tabs(["📋 Salary Prediction", "📊 Insights", "ℹ️ About Project"])

# ======================================================================================
# TAB 1: Salary Prediction Form
# ======================================================================================
with tab1:
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    with st.form("salary_form"):
        st.subheader("Employee Details")

        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=18, max_value=80, value=30)
            gender = st.selectbox("Gender", options=label_encoders["Gender"].classes_)
            years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)
        with col2:
            education_level = st.selectbox("Education Level", options=label_encoders["Education Level"].classes_)
            job_title = st.selectbox("Job Title", options=label_encoders["Job Title"].classes_)

        submitted = st.form_submit_button("Predict Salary")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- Prediction Result Section ---
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

        st.success(f"💰 **Predicted Annual Salary:** ₹{predicted_salary * 13:,.0f}")

        st.metric(label="Predicted Monthly Salary", value=f"₹{predicted_salary:,.0f}")
        st.metric(label="Experience Level", value="Senior" if years_of_experience > 10 else "Mid-Level" if years_of_experience > 3 else "Entry-Level")
        st.progress(min(years_of_experience / 40, 1.0))

        st.markdown("---")
        st.image(eval_plot, caption="Model Evaluation: Actual vs. Predicted Salaries", use_container_width=True)

# ======================================================================================
# TAB 2: Insights
# ======================================================================================
with tab2:
    st.subheader("📈 Sample Data Insights")
    st.info("Visual insights about features affecting salary")

    # Example mini data visualization (fake but for demo purpose)
    data = {
        "Education Level": ["High School", "Bachelor’s", "Master’s", "PhD"],
        "Avg Salary (₹)": [350000, 700000, 1200000, 1800000]
    }
    df = pd.DataFrame(data)
    st.bar_chart(df.set_index("Education Level"))

    exp_df = pd.DataFrame({
        "Years of Experience": [0, 5, 10, 15, 20, 25],
        "Estimated Salary (₹)": [200000, 500000, 900000, 1200000, 1500000, 1800000]
    })
    st.line_chart(exp_df.set_index("Years of Experience"))

# ======================================================================================
# TAB 3: About Project
# ======================================================================================
with tab3:
    st.subheader("ℹ️ Project Summary")
    st.write("""
    **Project Title:** Salary Prediction System using Machine Learning  
 
    **Objective:**  
    To predict an employee's salary based on demographic and professional attributes such as age, gender, education, and experience.

    **Features of this App:**
    - User-friendly interface with clean design  
    - Machine learning model integrated using `joblib`  
    - Dynamic visualizations and metrics  
    - Cached loading for performance  
    - Insight dashboard for data exploration  

    **Tools Used:** Streamlit, Pandas, Scikit-learn, Joblib  
    **Dataset Source:** Internal salary dataset (pre-trained model)
    """)









