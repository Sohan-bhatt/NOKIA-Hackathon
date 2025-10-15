import streamlit as st
import pandas as pd
import os
from models.model_training import preprocess_alarm_data, train_all_models, save_all_models

# --- Page config ---
st.set_page_config(page_title="ML Modeling", page_icon="🤖", layout="wide")

# Load custom CSS
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Check for raw input ---
if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
    st.warning("No raw data available. Please upload or load sample data from the Data Upload page.")
    st.stop()

# Header
st.markdown(
    """
    <div class="header">
        <div class="logo">🤖</div>
        <div class="title-container">
            <h1 class="main-title">ML Model Training</h1>
            <p class="subtitle">Train all models for multi-step network alarm prediction</p>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)

# --- Fresh copy of raw data ---
df = st.session_state.raw_data.copy()

# --- Ensure required numeric columns exist ---
required_numeric = ['Life Span (minutes)', 'Number Of Occurrences', 'Is Service Affecting', 'is_active']
for col in required_numeric:
    if col not in df.columns:
        if col == "is_active":
            cleared_candidates = [c for c in df.columns if "cleared" in c.lower()]
            if cleared_candidates:
                df[col] = df[cleared_candidates[0]].isnull().astype(int)
            else:
                df[col] = 0
        else:
            df[col] = 0 if col != "Number Of Occurrences" else 1

# --- Modeling Section ---
st.markdown("## 🧠 Train All Models (Chained Pipeline)")

if st.button("Train All Models"):
    with st.spinner("Preprocessing and training all models..."):
        try:
            # 1. Preprocess raw copy safely
            df_clean, df_time, encoders = preprocess_alarm_data(df)

            # 2. Train all models
            model_dict = train_all_models(df_clean, df_time)

            # 3. Save models
            save_all_models(model_dict, encoders, out_dir="models")

            # 4. Store in session
            st.session_state.model_bundle = (model_dict, encoders)
            st.session_state.preprocessed_data = df_clean

            # Feedback
            st.success("✅ All models trained and saved! Ready for predictions.")
            st.markdown("### 🧾 First rows of processed training data:")
            st.dataframe(df_clean.head(8), use_container_width=True)

        except KeyError as e:
            st.error(f"Missing required column: `{str(e)}`.\nAvailable columns: {list(df.columns)}")
        except Exception as ex:
            st.error(f"Unexpected error during training: {ex}")

# --- Trained Model Summary ---
if 'model_bundle' in st.session_state:
    st.markdown("## 📊 Trained Model Summary")
    st.markdown("""
    - ✅ Alarm Type Classifier  
    - ✅ Probable Cause Classifier  
    - ✅ Duration Regressor  
    - ✅ Severity Classifier  
    - ✅ Next Alarm Timestamp Regressor  
    """)
    st.info("Switch to the Predictions page to try multi-output forecasting.")

# --- Navigation ---
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("[← Back to EDA](/pages/2_eda.py)")
with col2:
    st.markdown("[Continue to Predictions →](/pages/4_predictions.py)")
