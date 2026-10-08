import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Analyzer", 
    page_icon="💳", 
    layout="wide"
)

# Custom Aesthetic CSS Injection with Gradient Background and Transitions
st.markdown("""
<style>
    /* Completely remove Streamlit top header, deploy button, and menu */
    header[data-testid="stHeader"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Global Background with Smooth Gradient Transition */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        background-attachment: fixed;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
        transition: background 0.5s ease-in-out;
    }
    
    /* Remove default padding at the very top of the page */
    .block-container {
        padding-top: 2rem !important;
    }
    
    /* Footer visibility */
    footer {visibility: hidden;}

    /* Sidebar Styling with matching tone */
    section[data-testid="stSidebar"] {
        background-color: #171c30;
        border-right: 1px solid #2d2b55;
        transition: background-color 0.3s ease;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }

    /* Custom Cards / Containers with Glassmorphism Effect */
    .metric-card {
        background: rgba(30, 27, 75, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(139, 92, 246, 0.2);
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        margin-bottom: 15px;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(139, 92, 246, 0.5);
        transform: translateY(-2px);
    }

    /* Headers with vibrant accent colors */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600;
    }
    
    p, span, label {
        color: #cbd5e1;
    }

    /* Custom Violet-Indigo Buttons with Smooth Hover Transition */
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9 0%, #4338ca 100%);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Load Model & Features
@st.cache_resource
def load_assets():
    model = joblib.load('loan_decision_tree_model.pkl')
    model_features = joblib.load('model_features.pkl')
    return model, model_features

model, model_features = load_assets()

# Header Section
st.title("💳 Executive Credit Risk & Loan Assessment")
st.markdown("<p style='color: #a5b4fc; font-size: 1.1rem;'>An indigo-violet themed decision-support system powered by interpretable Machine Learning.</p>", unsafe_allow_html=True)
st.divider()

# Sidebar Inputs
st.sidebar.header("📝 Applicant Profile")
st.sidebar.markdown("Configure financial parameters below:")

applicant_age = st.sidebar.slider("Applicant Age", 18, 90, 32)
annual_income = st.sidebar.number_input("Annual Income ($)", min_value=1000, max_value=2000000, value=65000, step=1000)
requested_loan_amount = st.sidebar.number_input("Requested Loan Amount ($)", min_value=500, max_value=500000, value=15000, step=500)
interest_rate = st.sidebar.slider("Loan Interest Rate (%)", 4.0, 30.0, 11.5, 0.1)

# Derived feature
loan_to_income_ratio = requested_loan_amount / annual_income if annual_income > 0 else 0

# Main Panel Layout
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.subheader("📊 Financial Overview")
    
    # Styled Card Container using Markdown
    st.markdown("""
    <div class="metric-card">
        <h4 style="margin-top: 0; color: #c4b5fd;">Key Financial Indicators</h4>
    """, unsafe_allow_html=True)
    
    m1, m2 = st.columns(2)
    m1.metric("Annual Income", f"${annual_income:,.0f}")
    m2.metric("Requested Amount", f"${requested_loan_amount:,.0f}")
    
    m3, m4 = st.columns(2)
    m3.metric("Interest Rate", f"{interest_rate}%")
    m4.metric("Debt-to-Income Ratio", f"{loan_to_income_ratio:.2f}")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.subheader("🎯 Risk Evaluation")
    st.markdown("<p style='color: #a5b4fc;'>Run the decision algorithm against the structured applicant profile.</p>", unsafe_allow_html=True)
    
    # Construct input dataframe
    input_data = {
        'applicant_age': applicant_age,
        'annual_income': annual_income,
        'requested_loan_amount': requested_loan_amount,
        'interest_rate': interest_rate,
        'loan_to_income_ratio': loan_to_income_ratio
    }
    input_df = pd.DataFrame([input_data])
    
    for col in model_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[model_features]

    # Prediction Action inside a styled container
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Analyze Credit Risk", use_container_width=True):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.divider()
        
        if prediction[0] == 1:
            st.error("⚠️ **High Risk Detected: Loan Rejected**")
            st.markdown(f"Model estimates a **{prediction_proba[0][1]*100:.1f}% probability of default**.")
        else:
            st.success("✅ **Low Risk: Loan Approved**")
            st.markdown(f"Model estimates a **{prediction_proba[0][0]*100:.1f}% confidence** in successful repayment.")