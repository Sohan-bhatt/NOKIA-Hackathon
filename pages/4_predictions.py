# pred.py

import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
from models.model_training import chained_predict, batch_chained_predict, load_all_models

# --- Page config ---
st.set_page_config(page_title="Alarm Predictions", page_icon="🔮", layout="wide")

# Load custom CSS
if os.path.exists("assets/style.css"):
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Robust model loading ---
if 'model_bundle' not in st.session_state or st.session_state.model_bundle is None:
    try:
        model_dict, encoders = load_all_models("models")
        st.session_state.model_bundle = (model_dict, encoders)
    except Exception as e:
        st.warning("No trained models available. Please train models on the ML Modeling page.")
        st.markdown("[← Go to ML Modeling](/pages/3_modeling.py)")
        st.stop()

model_dict, encoders = st.session_state.model_bundle

# --- Data in session_state (preprocessed + encoded) ---
if 'preprocessed_data' not in st.session_state or st.session_state.preprocessed_data is None:
    st.warning("No preprocessed data found. Please train models first on the ML Modeling page.")
    st.stop()
df = st.session_state.preprocessed_data.copy()

# --- Infer which column represents “Site Name” (encoded) ---
site_col = None
for c in ["Site Name", "site_id", "site", "NODE", "Alarmed Object Name"]:
    if c in df.columns:
        site_col = c
        break
if site_col is None:
    st.error("No site column found in your data. Please check your dataset.")
    st.stop()

# --- Header ---
st.markdown("""
    <div class="header">
        <div class="logo">🔮</div>
        <div class="title-container">
            <h1 class="main-title">Alarm Predictions</h1>
            <p class="subtitle">Multi-step prediction of next alarm, cause, duration, severity, and more.</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Prediction Configuration ---
st.markdown("## ⚙️ Prediction Configuration")

prediction_type = st.radio(
    "What do you want to predict?",
    ["Chained Prediction for Top 10", "Next Alarm for a Site", "Alarms for All Sites"],
    horizontal=True
)

features = model_dict['input_features']


if prediction_type == "Chained Prediction for Top 10":
    st.markdown("Top 10 chained predictions from test set (last 10 rows):")
    X = df[features['alarm_type']].tail(10)
    results_df = batch_chained_predict(X, model_dict, encoders)
    st.dataframe(results_df, use_container_width=True)

    st.markdown("### Download as CSV")
    st.download_button("Download Predictions", results_df.to_csv(index=False), file_name="chained_predictions.csv")


elif prediction_type == "Next Alarm for a Site":
    # 1) We need the “Site Name” encoder to decode back to human‐readable labels
    site_encoder = encoders.get("Site Name")
    if site_encoder is None:
        st.error("Site encoder not found. Make sure 'Site Name' was encoded during training.")
        st.stop()

    # 2) Build a mapping: encoded_value → original_label
    site_mapping = dict(enumerate(site_encoder.classes_))
    label_to_encoded = {v: k for k, v in site_mapping.items()}

    # 3) Let user pick a human‐readable site label
    selected_site_label = st.selectbox("Select Site", list(label_to_encoded.keys()))
    encoded_value = label_to_encoded[selected_site_label]

    # 4) Fetch the last‐row for that site (encoded)
    row = df[df['Site Name'] == encoded_value].tail(1)

    if row.empty:
        st.info("No data available for this site.")
    else:
        result = chained_predict(row, model_dict, encoders)
        st.markdown(f"## 🔮 Prediction for Site: `{selected_site_label}`")

        # --- Plotly‐based Card Layout ---
        
    fig = go.Figure()

    # Alarm Type
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=1,
        number={"prefix": result['Predicted Alarm Type'], "font": {"size": 28, "color": "#636EFA"}},
        title={"text": "<b>Alarm Type</b>", "font": {"size": 16}},
        domain={"row": 0, "column": 0}
    ))

    # Cause
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=1,
        number={"prefix": result['Predicted Probable Cause'], "font": {"size": 28, "color": "#EF553B"}},
        title={"text": "<b>Cause</b>", "font": {"size": 16}},
        domain={"row": 0, "column": 1}
    ))

    # Severity
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=1,
        number={"prefix": result['Predicted Severity'], "font": {"size": 28, "color": "#00CC96"}},
        title={"text": "<b>Severity</b>", "font": {"size": 16}},
        domain={"row": 0, "column": 2}
    ))

    # Confidence
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=result['Alarm Type Confidence'],
        number={"suffix": " 🧠", "font": {"color": "#AB63FA", "size": 24}},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#AB63FA"},
            "steps": [
                {"range": [0, 0.5], "color": "#F3E5F5"},
                {"range": [0.5, 0.8], "color": "#D1C4E9"},
                {"range": [0.8, 1.0], "color": "#9575CD"},
            ],
        },
        title={"text": "<b>Confidence</b>", "font": {"size": 16}},
        domain={"row": 0, "column": 3}
    ))

    fig.update_layout(
        grid={"rows": 1, "columns": 4, "pattern": "independent"},
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


elif prediction_type == "Alarms for All Sites":
    # Get every unique encoded site ID
    unique_sites = df[site_col].unique()
    st.markdown(f"#### Predictions for all {len(unique_sites)} sites (latest alarm record for each):")

    pred_rows = []
    for site_encoded in unique_sites:
        row = df[df[site_col] == site_encoded].tail(1)
        if not row.empty:
            pred = chained_predict(row, model_dict, encoders)
            # Place the encoded site_id into the results, so we can decode it later
            pred['Site'] = int(site_encoded)
            pred_rows.append(pred)

    if len(pred_rows) == 0:
        st.info("No rows to predict.")
    else:
        results_df = pd.DataFrame(pred_rows)

        # Decode all relevant columns
        decoded_df = decode_columns(results_df, encoders)

        st.dataframe(decoded_df, use_container_width=True)
        st.markdown("### Download as CSV")
        st.download_button("Download Predictions", decoded_df.to_csv(index=False), file_name="chained_predictions.csv")


def decode_columns(df_predictions: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """
    Takes a DataFrame with columns:
      - 'Site'                     (encoded integer)
      - 'Predicted Alarm Type'     (string & already decoded by chained_predict)
      - 'Predicted Probable Cause' (string & already decoded)
      - 'Predicted Severity'       (string & already decoded)
      - potentially other numeric/confidence columns
    
    We only need to decode 'Site' here. The 'Predicted ...' fields are already decoded
    inside chained_predict and are strings. If you ever add any other encoded‐integer column
    to this DataFrame, extend the mapping below.
    """
    df_copy = df_predictions.copy()

    # 1) Decode 'Site' using the "Site Name" encoder
    if 'Site' in df_copy.columns and 'Site Name' in encoders:
        site_le = encoders['Site Name']
        df_copy['Site'] = site_le.inverse_transform(df_copy['Site'].astype(int))

    # 2) The chained_predict() already returned human‐readable strings for:
    #    'Predicted Alarm Type', 'Predicted Probable Cause', 'Predicted Severity'.
    #    If for some reason you stored them as integers, you could use:
    #      encoders['Alarm Type'].inverse_transform(...)
    #    But here we assume they are already strings.

    return df_copy


# --- Navigation footer ---
st.markdown("---")
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown("[← Back to ML Modeling](/pages/3_modeling.py)")
with col2:
    st.markdown("[Continue to Visualizations →](/pages/5_visualizations.py)")
