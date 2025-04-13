import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import streamlit as st

# Custom title section with blue background
st.markdown(
    """
    <div style="background-color:#1f77b4;padding:20px;border-radius:10px">
    <h1 style="color:white;text-align:center;">Credit Loan Eligibility Predictor</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Description
st.markdown(
    """
    <p style="font-size:16px;color:#333;text-align:center;margin-top:10px;">
    This app predicts whether a loan applicant is eligible for a loan 
    based on various personal and financial characteristics.
    </p>
    """,
    unsafe_allow_html=True
)

# Add some custom CSS for input styling
st.markdown("""
    <style>
        /* Custom Background */
        body {
            background: linear-gradient(135deg, #f0f4f8, #c2c7d0);
            font-family: 'Arial', sans-serif;
        }
        
        .title {
            color: #1f77b4;
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 20px;
        }

        .subheader {
            text-align: center;
            color: #333;
            font-size: 1.5em;
        }

        .input-container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
        }

        .input-label {
            font-weight: bold;
            color: #1f77b4;
        }

        .stButton button {
            background-color: #1f77b4;
            color: white;
            padding: 12px 24px;
            font-size: 1.1em;
            border-radius: 5px;
            border: none;
        }

        .stButton button:hover {
            background-color: #155a8a;
        }

        .result-text {
            font-size: 1.2em;
            color: #28a745;
            text-align: center;
            margin-top: 20px;
        }

        .image-container {
            text-align: center;
            margin-top: 30px;
        }

        .image-container img {
            border-radius: 15px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .stTextInput, .stNumberInput, .stSelectbox, .stTextArea {
        background-color: #f5f5f5;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
        font-size: 14px;
        color: #333;
    }
    .stTextInput:hover, .stNumberInput:hover, .stSelectbox:hover, .stTextArea:hover {
        border-color: #1f77b4;
    }
    .stSelectbox select, .stNumberInput input {
        background-color: #ffffff;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)

# Load the pre-trained model
try:
    with open("models/RFmodel.pkl", "rb") as rf_pickle:
        rf_model = pickle.load(rf_pickle)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'models/RFmodel.pkl' exists.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

# --- Form section ---
with st.form("user_inputs"):
    st.subheader("Loan Applicant Details")
    
    # Create columns for the form
    col1, col2= st.columns(2)
    
    with col1:
        Gender = st.selectbox("Gender", options=["Male", "Female"])
        Married = st.selectbox("Marital Status", options=["Yes", "No"])
        ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0, step=1000)
        Loan_Amount_Term = st.selectbox("Loan Amount Term (Months)", options=["360", "180", "240", "120", "60"])
        Self_Employed = st.selectbox("Self Employed", options=["Yes", "No"])

        
    with col2:
        Dependents = st.selectbox("Number of Dependents", options=["0", "1", "2", "3+"])
        Education = st.selectbox("Education Level", options=["Graduate", "Not Graduate"])
        CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0, step=1000)
        Credit_History = st.selectbox("Credit History", options=["1", "0"])
        LoanAmount = st.number_input("Loan Amount", min_value=0, step=1000)
    Property_Area = st.selectbox("Property Area", options=["Urban", "Semiurban", "Rural"])
        

    submitted = st.form_submit_button("Predict Loan Eligibility")

# --- Prediction section ---
if submitted:
    try:
        Gender_Male = 0 if Gender == "Female" else 1
        Gender_Female = 1 if Gender == "Female" else 0

        Married_Yes = 1 if Married == "Yes" else 0
        Married_No = 1 if Married == "No" else 0

        Dependents_0 = 1 if Dependents == "0" else 0
        Dependents_1 = 1 if Dependents == "1" else 0
        Dependents_2 = 1 if Dependents == "2" else 0
        Dependents_3 = 1 if Dependents == "3+" else 0

        Education_Graduate = 1 if Education == "Graduate" else 0
        Education_Not_Graduate = 1 if Education == "Not Graduate" else 0

        Self_Employed_Yes = 1 if Self_Employed == "Yes" else 0
        Self_Employed_No = 1 if Self_Employed == "No" else 0

        Property_Area_Rural = 1 if Property_Area == "Rural" else 0
        Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
        Property_Area_Urban = 1 if Property_Area == "Urban" else 0

        Loan_Amount_Term = int(Loan_Amount_Term)
        Credit_History = int(Credit_History)

        prediction_input = [[
            ApplicantIncome, CoapplicantIncome, LoanAmount,
            Loan_Amount_Term, Credit_History, Gender_Female, Gender_Male,
            Married_No, Married_Yes, Dependents_0, Dependents_1,
            Dependents_2, Dependents_3, Education_Graduate,
            Education_Not_Graduate, Self_Employed_No, Self_Employed_Yes,
            Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
        ]]

        new_prediction = rf_model.predict(prediction_input)

        st.markdown("Prediction Result:")
        if new_prediction[0] == 'Y':
            st.success("You are eligible for the loan!")
        else:
            st.error("Sorry, you are not eligible for the loan.")

        # Try to display the image
        try:
            st.image("feature_importance.png")
        except FileNotFoundError:
            st.warning("Feature importance image not found.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Feature explanation ---
st.markdown(
    """
    <div style="background-color:#f0f2f6;padding:10px 20px;border-radius:10px;margin-top:30px">
    <h4 style="color:#1f77b4;"> How does this work?</h4>
    <p style="color:#333;">
    We used a machine learning <strong>Random Forest</strong> model to predict your eligibility.
    The features used in this prediction are ranked by relative importance below:
    </p>
    </div>
    """,
    unsafe_allow_html=True
)