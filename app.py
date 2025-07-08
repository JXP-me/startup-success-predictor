import streamlit as st
import pandas as pd
import joblib

# 🚀 Load the trained model and feature columns
model = joblib.load("models/logistic_model.pkl")
model_columns = joblib.load("models/columns.pkl")

st.set_page_config(page_title="Startup Success Predictor", layout="centered")
st.title("🚀 Startup Success Predictor")
st.write("Enter startup features below to predict whether it will succeed or not.")

# 🧾 Form to input features
with st.form("prediction_form"):
    st.subheader("🔧 Startup Details")

    age_first_funding_year = st.number_input("Age at First Funding Year", min_value=0.0, value=1.0)
    age_last_funding_year = st.number_input("Age at Last Funding Year", min_value=0.0, value=1.0)
    age_first_milestone_year = st.number_input("Age at First Milestone", min_value=0.0, value=1.0)
    age_last_milestone_year = st.number_input("Age at Last Milestone", min_value=0.0, value=1.0)
    relationships = st.number_input("Relationships", min_value=0)
    funding_rounds = st.number_input("Funding Rounds", min_value=0)
    funding_total_usd = st.number_input("Total Funding (USD)", min_value=0)
    milestones = st.number_input("Milestones", min_value=0)
    avg_participants = st.number_input("Average Participants", min_value=0.0, value=1.0)

    st.subheader("🌍 Location & Category")
    is_CA = st.checkbox("Is California (CA)?")
    is_NY = st.checkbox("Is New York (NY)?")
    is_MA = st.checkbox("Is Massachusetts (MA)?")
    is_TX = st.checkbox("Is Texas (TX)?")
    is_otherstate = st.checkbox("Other State?")

    is_software = st.checkbox("Software?")
    is_web = st.checkbox("Web?")
    is_mobile = st.checkbox("Mobile?")
    is_enterprise = st.checkbox("Enterprise?")
    
    submitted = st.form_submit_button("Predict")

# 🎯 Predict once form is submitted
if submitted:
    # Create input dictionary
    user_input = {
        'age_first_funding_year': age_first_funding_year,
        'age_last_funding_year': age_last_funding_year,
        'age_first_milestone_year': age_first_milestone_year,
        'age_last_milestone_year': age_last_milestone_year,
        'relationships': relationships,
        'funding_rounds': funding_rounds,
        'funding_total_usd': funding_total_usd,
        'milestones': milestones,
        'avg_participants': avg_participants,
        'is_CA': int(is_CA),
        'is_NY': int(is_NY),
        'is_MA': int(is_MA),
        'is_TX': int(is_TX),
        'is_otherstate': int(is_otherstate),
        'is_software': int(is_software),
        'is_web': int(is_web),
        'is_mobile': int(is_mobile),
        'is_enterprise': int(is_enterprise),
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([user_input])

    # 🧠 Add any missing columns (that model was trained with)
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # 📊 Reorder columns to match model
    input_df = input_df[model_columns]

    # ✅ Make prediction
    prediction = model.predict(input_df)[0]

    st.markdown("---")
    st.subheader("📈 Prediction Result:")
    if prediction == 1:
        st.success("🎉 The startup is likely to SUCCEED!")
    else:
        st.error("⚠️ The startup is likely to FAIL.")

