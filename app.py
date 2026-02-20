import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --- Configuration ---
st.set_page_config(
    page_title="Loan Default Prediction (Random Forest)",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Load Model & Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('RF.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading model artifacts: {e}")
        return None, None, None

model, scaler, feature_names = load_artifacts()

if model is None:
    st.stop()

# Title and Description 
st.title('🏦 Loan Default Risk Predictor')
st.markdown("""
This application predicts the likelihood of a loan applicant defaulting on their loan using a trained Random Forest classifier.
Adjust the parameters in the sidebar and main panel to see the risk assessment.
""")

#  Sidebar Inputs 
st.sidebar.header("Applicant Profile")

# Gender
gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Others'])

# Marital Status
marital_status = st.sidebar.selectbox(
    'Marital Status',
    ['Single', 'Married', 'Divorced', 'Widowed']
)

# Education Level
education_level = st.sidebar.selectbox(
    'Education Level',
    ['High School', "Bachelor's", "Master's", 'PhD', 'Other']
)

# Employment Status
employment_status = st.sidebar.selectbox(
    'Employment Status',
    ['Employed', 'Self_Employed', 'Unemployed', 'Student', 'Retired']
)

# Loan Purpose
loan_purpose = st.sidebar.selectbox(
    'Loan Purpose',
    ['Business', 'Car', 'Debt_consolidation', 'Education', 'Home', 'Medical', 'Other', 'Vacation']
)

# Loan Grade (Risk Level)
grade_subgrade = st.sidebar.selectbox(
    'Loan Grade (Risk Level)',
    ['A1','A2','A3','A4','A5',
     'B1','B2','B3','B4','B5',
     'C1','C2','C3','C4','C5',
     'D1','D2','D3','D4','D5',
     'E1','E2','E3','E4','E5',
     'F1','F2','F3','F4','F5']
)

# --- Main Panel Inputs ---
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input(
        'Annual Income ($)',
        min_value=0.0,
        max_value=10_000_000.0,
        value=50_000.0,
        step=1000.0,
        format="%.2f",
        help="The reliable yearly income of the applicant."
    )

    loan_amount = st.number_input(
        'Loan Amount ($)',
        min_value=0.0,
        max_value=1_000_000.0,
        value=10_000.0,
        step=500.0,
        format="%.2f",
        help="The total amount of money requested for the loan."
    )

    interest_rate = st.number_input(
        'Interest Rate (%)',
        min_value=0.0,
        max_value=100.0,
        value=12.5,
        step=0.1,
        format="%.2f",
        help="The annual interest rate for the loan."
    )

with col2:
    debt_to_income_ratio = st.number_input(
        'Debt-to-Income Ratio (DTI)',
        min_value=0.0,
        max_value=10.0,
        value=0.30,
        step=0.01,
        format="%.2f",
        help="Calculated by dividing total recurring monthly debt by gross monthly income."
    )

    credit_score = st.slider(
        'Credit Score',
        min_value=300,
        max_value=850,
        value=650,
        help="A numerical expression based on a level analysis of a person's credit files."
    )


# Prediction Logic 
if st.button('🚀 Analyze Risk', type="primary", use_container_width=True):
    
    # 1. Prepare Features
    
    # Mapping - Must match notebook exactly
    education_level_map = {
        'High School': 1, "Bachelor's": 2, "Master's": 3, 'PhD': 4, 'Other': 5
    }
    # Note: Grade mapping logic from notebook is sequential 1-30
    grade_order = ['A1','A2','A3','A4','A5',
                   'B1','B2','B3','B4','B5',
                   'C1','C2','C3','C4','C5',
                   'D1','D2','D3','D4','D5',
                   'E1','E2','E3','E4','E5',
                   'F1','F2','F3','F4','F5']
    grade_subgrade_map = {grade: i+1 for i, grade in enumerate(grade_order)}
    
    edu_val = education_level_map.get(education_level, 0)
    grade_val = grade_subgrade_map.get(grade_subgrade, 0)
    
    # One-Hot Encoding Construction
    # We create a dictionary with all expected features initialized to 0
    input_dict = {f: 0 for f in feature_names}
    
    # Fill Numeric Features
    # Note: Check feature names for exact spelling/case from notebook run
    # Assuming notebook used: annual_income, debt_to_income_ratio, credit_score, loan_amount, interest_rate
    # And: education_level, grade_subgrade (after rename)
    
    input_dict['annual_income'] = annual_income
    input_dict['debt_to_income_ratio'] = debt_to_income_ratio
    input_dict['credit_score'] = credit_score
    input_dict['loan_amount'] = loan_amount
    input_dict['interest_rate'] = interest_rate
    input_dict['education_level'] = edu_val
    input_dict['grade_subgrade'] = grade_val
    
    # Fill Categorical Dummies
    # Notebook: pd.get_dummies(..., drop_first=True)
    # The feature names will be like 'gender_Male', 'gender_Other'
    
    # Gender
    # If Female dropped (ref), gender_Male=1 if Male, gender_Other=1 if Others
    if gender == 'Male':
        if 'gender_Male' in input_dict: input_dict['gender_Male'] = 1
    elif gender == 'Others':
        if 'gender_Others' in input_dict: input_dict['gender_Others'] = 1 # Check exact name
        elif 'gender_Other' in input_dict: input_dict['gender_Other'] = 1

    # Marital Status
    # Ref likely Divorced (alphabetical 1st). 
    # Terms: Married, Single, Widowed
    ms_key = f"marital_status_{marital_status}"
    if ms_key in input_dict: input_dict[ms_key] = 1
    
    # Employment
    # Ref likely Employed.
    # Terms: Retired, Self_Employed, Student, Unemployed
    # Notebook might escape chars using replace(' ', '_')
    # Try direct mapping first
    
    # Feature names form notebook show 'employment_status_Self-employed' (hyphen)
    # App input is 'Self_Employed' (underscore)
    
    # We construct potential keys and check if they exist in feature_names
    emp_keys_to_try = [
        f"employment_status_{employment_status}",
        f"employment_status_{employment_status.replace('_', '-')}", # Self_Employed -> Self-Employed
        f"employment_status_{employment_status.replace('_', ' ')}"  # Self_Employed -> Self Employed
    ]
    
    for k in emp_keys_to_try:
        if k in input_dict:
            input_dict[k] = 1
            break

    # Loan Purpose
    # Ref likely Business (alphabetical 1st?)
    # Terms: Car, Debt_consolidation, Education, Home, Medical, Other, Vacation
    # Feature names often use spaces or original CSV values
    lp_keys_to_try = [
        f"loan_purpose_{loan_purpose}",
        f"loan_purpose_{loan_purpose.replace('_', ' ')}", # Debt_consolidation -> Debt consolidation
    ]
    
    for k in lp_keys_to_try:
        if k in input_dict:
            input_dict[k] = 1
            break
            
    # Create Input DataFrame
    # IMPORTANT: Columns must be in exact order of feature_names as used during training
    # We reindex to be safe
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names] 
    
    # 2. Scale
    try:
        input_scaled = scaler.transform(input_df)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.stop()
        
    # 3. Predict
    # Model now predicts 1=Default, 0=Paid Back
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Determine probability of default (Class 1)
    prob_default = prediction_proba[1] 
    prob_payback = prediction_proba[0]

    #  Display Results 
    st.divider()
    
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    with col_kpi1:
        st.metric("Loan Status Prediction", "Default" if prob_default > 0.5 else "Fully Paid", delta_color="inverse")
    with col_kpi2:
        st.metric("Default Probability", f"{prob_default:.1%}")
    with col_kpi3:
        st.metric("Repayment Probability", f"{prob_payback:.1%}")

    st.subheader("Risk Analysis")
    st.progress(int(prob_default * 100))
    
    if prob_default > 0.65:
        st.error(f"**High Risk**: There is a high chance ({prob_default:.1%}) this loan will default.")
    elif prob_default > 0.35:
        st.warning(f"**Moderate Risk**: Default probability is {prob_default:.1%}. Carefully review credit history.")
    else:
        st.success(f"**Low Risk**: High chance ({prob_payback:.1%}) of repayment.")

st.markdown("---")
st.caption("Model Version: 1.0.1 | Trained on Historical Data")
