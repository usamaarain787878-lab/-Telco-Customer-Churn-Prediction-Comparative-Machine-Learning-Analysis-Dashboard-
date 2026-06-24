import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import re
import subprocess  # Added for auto-training fallback

# App utils and backend interfaces
from utils import render_model_training_section, render_model_comparison_section

# --- DYNAMIC PATH CONFIGURATION (Fixes Cloud File Path Issues) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Dark and professional styling configuration
st.set_page_config(page_title="Telco Churn Analytics Ecosystem", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    h1, h2, h3 { color: #00d2ff !important; font-family: 'Helvetica Neue', sans-serif; }
    .stButton>button { background-color: #00d2ff; color: black; border-radius: 6px; font-weight: bold; width: 100%; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)


# --- AUTOMATIC TRAINING FALLBACK TRIGGER ---
# Agar core files missing hain to cloud par khud hi training run ho jaye
metrics_csv_path = os.path.join(MODELS_DIR, 'model_comparison_metrics.csv')
if not os.path.exists(metrics_csv_path):
    training_script = os.path.join(BASE_DIR, 'train_models.py')
    if os.path.exists(training_script):
        try:
            # Silent background trigger so the app doesn't break on first load
            subprocess.run(['python', training_script], check=True)
        except Exception as e:
            pass


# Helper function to safely load resources with dual-path fallback
@st.cache_resource
def load_resources():
    primary_model_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    fallback_model_path = os.path.join(MODELS_DIR, 'logistic_regression_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')            
    feature_names_path = os.path.join(MODELS_DIR, 'feature_names.pkl') 
    
    model = None
    scaler = None
    feature_names = None
    
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        
    if os.path.exists(feature_names_path):
        feature_names = joblib.load(feature_names_path)
        
    if os.path.exists(primary_model_path):
        try:
            model = joblib.load(primary_model_path)
        except Exception as e:
            st.error(f"Error loading champion model: {e}")
            model = None
            
    if model is None and os.path.exists(fallback_model_path):
        try:
            model = joblib.load(fallback_model_path)
        except Exception as e:
            st.error(f"Error loading fallback model: {e}")
            model = None
            
    return model, scaler, feature_names

model, scaler, feature_names_data = load_resources()


# --- OPTIMIZED DATA LOADING FUNCTION WITH MULTI-PATH FALLBACK ---
@st.cache_data
def load_and_clean_data():
    paths_to_check = [
        os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv'), 
        os.path.join(BASE_DIR, 'Telco-Customer-Churn.csv'), 
        os.path.join(BASE_DIR, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    ]
    
    path_found = None
    for path in paths_to_check:
        if os.path.exists(path):
            path_found = path
            break
            
    if not path_found:
        if os.path.exists(BASE_DIR):
            csv_files = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('.csv') and 'churn' in f.lower()]
            if csv_files:
                path_found = csv_files[0]

    if path_found:
        try:
            df = pd.read_csv(path_found)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
            return df
        except Exception as e:
            st.error(f"Error reading CSV file ({path_found}): {e}")
            return None
            
    return None

# Load dataset once globally to share across layout pages safely
df_global = load_and_clean_data()

# Sidebar Navigation Control Center (Updated to include 6 required sections)
st.sidebar.title("🧭 Navigation Control")
app_mode = st.sidebar.radio("Jump to Dashboard Section:", 
    [
        "1. Home Page", 
        "2. Dataset Explorer", 
        "3. EDA Insights Canvas", 
        "4. Model Training Panel", 
        "5. Model Comparison Leaderboard", 
        "6. Live Client Prediction Form"
    ])

# ----------------- SECTION 1: HOME PAGE -----------------
if app_mode == "1. Home Page":
    st.markdown("<h1 style='font-size: 38px; font-weight: bold; margin-bottom: 0px;'>📊 Telco Customer Churn Prediction & Comparative ML Dashboard</h1>", unsafe_allow_html=True)
    st.subheader("Enterprise-Grade Machine Learning Solutions for Customer Retention Management")
    
    st.markdown("""
    ---
    ### 🎯 Business Objective & Context
    Acquiring new clients carries a significantly higher financial premium than retaining an active customer base. 
    When customers depart, businesses suffer immediate operational losses and recurring monthly contract revenue slips.
    
    By proactively mapping high-risk customer profiles early, marketing teams and customer success managers can 
    introduce precision retention interventions to stabilize revenue retention.
    
    ### 📂 Project Architecture Mapping:
    * **Phase 1-4:** Detailed EDA & Data Leakage Proof Preprocessing Pipelines
    * **Phase 5-7:** Optimization matrices built across 6 core machine learning models
    * **Phase 8-9:** Live interactive forecasting workspace engine deployed inside Streamlit Production Clouds
    """, unsafe_allow_html=True)

# ----------------- SECTION 2: DATASET EXPLORER -----------------
elif app_mode == "2. Dataset Explorer":
    st.title("🔍 Dataset Structural Explorer Workspace")
    
    if df_global is not None:
        st.success("✅ Dataset successfully located and loaded into memory!")
        st.subheader("Raw Data Sample View Matrix")
        st.dataframe(df_global.head(10), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Descriptive Statistical Analytics Summary")
            st.write(df_global.describe())
        with col2:
            st.subheader("Missing Attributes Log File")
            st.dataframe(pd.DataFrame(df_global.isnull().sum(), columns=["Missing Elements Count"]), use_container_width=True)
    else:
        st.error("⚠️ 'Telco-Customer-Churn.csv' data asset not detected inside the environment path setup.")
        st.info("💡 Solution: Apni CSV file ka naam 'Telco-Customer-Churn.csv' rakh kar use project folder ke andar save karein.")

# ----------------- SECTION 3: EDA INSIGHTS -----------------
elif app_mode == "3. EDA Insights Canvas":
    st.title("🎨 Interactive Exploratory Data Analysis Workspace")
    
    if df_global is not None:
        eda_feature = st.selectbox("Select Dimension to Dynamically Plot Against Churn:", ["Contract", "PaymentMethod", "InternetService", "StreamingTV"])
        
        fig = px.histogram(df_global, x=eda_feature, color="Churn", barmode="group",
                           title=f"Distribution Split Analysis: Churn vs {eda_feature}",
                           template="plotly_dark", color_discrete_sequence=["#2ca02c", "#d62728"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Please add the raw dataset inside the project directory to activate interactive plotting.")

# ----------------- SECTION 4: MODEL TRAINING PANEL -----------------
elif app_mode == "4. Model Training Panel":
    render_model_training_section(df_global)

# ----------------- SECTION 5: LEADERBOARD -----------------
elif app_mode == "5. Model Comparison Leaderboard":
    if os.path.exists(metrics_csv_path):
        render_model_comparison_section()
    else:
        st.warning("⚠️ Leaderboard metrics file missing.")
        st.info("💡 Solution: Please head over to '4. Model Training Panel' and train your models first to generate comparison charts.")

# ----------------- SECTION 6: LIVE CLIENT PREDICTION FORM -----------------
elif app_mode == "6. Live Client Prediction Form":
    st.title("🔮 Predictive Inference Control Center")
    
    live_model, live_scaler, live_features = load_resources()
    
    if live_model is not None and live_scaler is not None and live_features is not None:
        st.markdown("Input client profile parameters down below to gauge real-time deployment forecasting vector attributes:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Gender:", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen Categorization Status:", [0, 1])
            Partner = st.selectbox("Partner Status Profile:", ["Yes", "No"])
            Dependents = st.selectbox("Dependents Attributes:", ["Yes", "No"])
            tenure = st.slider("Tenure Months Range Matrix:", 0, 72, 12)
        with col2:
            PhoneService = st.selectbox("Phone Service Active Matrix:", ["Yes", "No"])
            MultipleLines = st.selectbox("Multiple Line Splits:", ["No phone service", "No", "Yes"])
            InternetService = st.selectbox("Internet Architecture Delivery Engine:", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("Online Security Firewall Layers:", ["No", "Yes", "No internet service"])
            OnlineBackup = st.selectbox("Online Backup Vault Arrays:", ["No", "Yes", "No internet service"])
            DeviceProtection = st.selectbox("Device Protection Insurance Coverage:", ["No", "Yes", "No internet service"])
        with col3:
            TechSupport = st.selectbox("Tech Support Concierge Routing:", ["No", "Yes", "No internet service"])
            StreamingTV = st.selectbox("Streaming TV Layout Matrices:", ["No", "Yes", "No internet service"])
            StreamingMovies = st.selectbox("Streaming Movie Playback Networks:", ["No", "Yes", "No internet service"])
            Contract = st.selectbox("Contract Subscription Class Framework:", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing Document Delivery:", ["Yes", "No"])
            PaymentMethod = st.selectbox("Payment Gateway Transaction Framework:", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.slider("Monthly Charges Billing Density Bounds ($):", 18.25, 118.75, 70.0)
            TotalCharges = st.number_input("Total Contract Accumulation Ledger Charges ($):", min_value=18.25, max_value=8684.80, value=840.0)
            
        raw_input = pd.DataFrame([{
            'gender': gender, 'SeniorCitizen': int(SeniorCitizen), 'Partner': Partner, 'Dependents': Dependents,
            'tenure': int(tenure), 'PhoneService': PhoneService, 'MultipleLines': MultipleLines,
            'InternetService': InternetService, 'OnlineSecurity': OnlineSecurity, 'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection, 'TechSupport': TechSupport, 'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies, 'Contract': Contract, 'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod, 'MonthlyCharges': float(MonthlyCharges), 'TotalCharges': float(TotalCharges)
        }])
        
        st.markdown("---")
        if st.button("🚀 Trigger Real-Time Machine Learning Diagnostic Assessment Inference"):
            try:
                input_encoded = pd.get_dummies(raw_input)
                full_input_df = pd.DataFrame(0, index=[0], columns=live_features)
                
                for col in input_encoded.columns:
                    if col in full_input_df.columns:
                        full_input_df[col] = input_encoded[col].values
                
                regex = re.compile(r"\[|\]|<", re.IGNORECASE)
                full_input_df.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in full_input_df.columns]
                full_input_df = full_input_df.astype(float)

                scaled_input = live_scaler.transform(full_input_df)
                prediction = live_model.predict(scaled_input)[0]
                
                if hasattr(live_model, "predict_proba"):
                    probability = live_model.predict_proba(scaled_input)[0][1]
                    prob_text = f" (Confidence Probability: {probability * 100:.2f}%)"
                else:
                    probability = None
                    prob_text = ""
                
                st.subheader("Diagnostic Engine Assessment Output Results:")
                is_churn = str(prediction).strip().lower() in ['1', 'yes', 'true'] or prediction == 1
                
                if is_churn:
                    st.error(f"🚨 ALERT: Target Profile identified as HIGH CHURN RISK Profile Matrix.{prob_text}")
                else:
                    stable_prob = f" (Confidence Probability: {(1 - probability) * 100:.2f}%)" if probability is not None else ""
                    st.success(f"💚 STABLE CLIENT: Profile identified as RETAINED ACTIVE ACCOUNT.{stable_prob}")
            
            except Exception as e:
                st.error("❌ **Pipeline Execution Error:** Preprocessing framework execution failed.")
                st.info(f"Technical Error Details: {e}")
    else:
        st.error("⚠️ Model file binaries or Pipeline Preprocessor assets not detected inside models/ directory paths.")
        st.info("💡 Solution: Pehle sidebar se '4. Model Training Panel' select karein aur kisi ek model ko train karein taake files automate ho jayein!")