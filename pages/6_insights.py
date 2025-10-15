# pages/6_insights.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- Page setup ---
st.set_page_config(page_title="Insights & Root Cause Analysis", page_icon="💡", layout="wide")
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Data check ---
if 'raw_data' not in st.session_state or st.session_state.raw_data is None:
    st.warning("No raw data available. Please upload or load sample data from the Data Upload page.")
    st.markdown("[← Go to Data Upload](/pages/1_data_upload.py)")
    st.stop()

df = st.session_state.raw_data.copy()


# --- Helper functions ---
def detect_column(df, candidates):
    return next((c for c in candidates if c in df.columns), None)

def clean_timestamp(col):
    return pd.to_datetime(
        col.astype(str)
           .str.replace(r"\s+\d{3}\s+[+-]\d{2}:\d{2}\s+GMT$", "", regex=True)
           .str.strip(), 
        errors='coerce'
    )

# --- Column detection ---
timestamp_col = detect_column(df, ["First Time Detected", "timestamp"])
site_col      = detect_column(df, ["Site Name", "site_id", "site"])
alarm_col     = detect_column(df, ["Alarm Type", "alarm_type", "alarm"])
severity_col  = detect_column(df, ["Previous Severity", "Severity"])
temperature_col = detect_column(df, ["temperature", "Temperature"])
cause_col=detect_column(df,["Probable Cause Merged","Probable Cause"])

# --- Parse timestamp if needed ---
if not np.issubdtype(df[timestamp_col].dtype, np.datetime64):
    df[timestamp_col] = clean_timestamp(df[timestamp_col])
df = df[df[timestamp_col].notna()]

# --- UI Header ---
st.markdown("""
    <div class="header">
        <div class="logo">💡</div>
        <div class="title-container">
            <h1 class="main-title">Insights & Root Cause Analysis</h1>
            <p class="subtitle">Explore frequent alarms, site behaviors, and severity origins</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

analysis_type = st.radio("🔎 Select Analysis Focus", 
                         ["Alarm Pattern Analysis", "Site-Specific Analysis", "Severity Root Cause"], 
                         horizontal=True)

# --- ALARM PATTERN ANALYSIS ---
if analysis_type == "Alarm Pattern Analysis":
    alarm_types = df[alarm_col].dropna().unique()
    selected_alarm = st.selectbox("Select Alarm Type", alarm_types)
    sub_df = df[df[alarm_col] == selected_alarm]

    st.subheader(f"📊 Alarm Insights: {selected_alarm}")
    st.metric("Total Occurrences", len(sub_df))
    st.metric("Affected Sites", sub_df[site_col].nunique())

    # Hourly Trend
    sub_df['hour'] = sub_df[timestamp_col].dt.hour
    fig = px.bar(sub_df.groupby('hour').size().reset_index(name="count"),
                 x="hour", y="count", title="Alarm Hourly Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Top sites
    top_sites = sub_df[site_col].value_counts().reset_index()
    top_sites.columns = ['Site', 'Count']
    fig = px.bar(top_sites.head(10), x='Site', y='Count', color='Count', title="Top Sites for Alarm")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Causes per sites
    top_sites = sub_df[cause_col].value_counts().reset_index()
    top_sites.columns = ['Root Cause', 'Count']
    fig = px.bar(top_sites.head(10), x='Root Cause', y='Count', color='Count', title="Top Causes for Alarm Type")
    st.plotly_chart(fig, use_container_width=True)

    # Severity distribution
    if severity_col in sub_df.columns:
        fig = px.pie(sub_df, names=severity_col, title="Severity Distribution for Alarm")
        st.plotly_chart(fig, use_container_width=True)

# --- SITE-SPECIFIC ANALYSIS ---
elif analysis_type == "Site-Specific Analysis":
    sites = df[site_col].dropna().unique()
    selected_site = st.selectbox("Select Site", sites)
    sub_df = df[df[site_col] == selected_site]

    st.subheader(f"🏗️ Site Analysis: {selected_site}")
    st.metric("Total Alarms", len(sub_df))
    st.metric("Unique Alarm Types", sub_df[alarm_col].nunique())

    if severity_col in sub_df.columns:
        criticals = sub_df[sub_df[severity_col].str.lower() == "critical"]
        st.metric("Critical Alarms", len(criticals))

    # Alarm type breakdown
    alarm_counts = sub_df[alarm_col].value_counts().reset_index()
    alarm_counts.columns = ['Alarm Type', 'Count']
    fig = px.bar(alarm_counts, x="Alarm Type", y="Count", color="Count", title=f"Alarm Types at Site {selected_site}")
    st.plotly_chart(fig, use_container_width=True)
    
    top_sites = sub_df[cause_col].value_counts().reset_index()
    top_sites.columns = ['Root Cause', 'Count']
    fig = px.bar(top_sites.head(10), x='Root Cause', y='Count', color='Count', title=f"Top Causes for Site {selected_site} ")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Temperature trends
    if temperature_col in sub_df.columns:
        temp_df = sub_df[[timestamp_col, temperature_col]].dropna()
        fig = px.line(temp_df, x=timestamp_col, y=temperature_col, title="Temperature Over Time")
        st.plotly_chart(fig, use_container_width=True)

# --- SEVERITY ROOT CAUSE ---
elif analysis_type == "Severity Root Cause":
    levels = df[severity_col].dropna().unique()
    selected_severity = st.selectbox("Select Severity", levels)
    sub_df = df[df[severity_col] == selected_severity]

    st.subheader(f"⚠️ Root Cause for Severity: {selected_severity}")
    st.metric("Total Occurrences", len(sub_df))
    st.metric("Distinct Alarm Types", sub_df[alarm_col].nunique())

    # Alarm Type distribution
    fig = px.histogram(sub_df, x=alarm_col, color=alarm_col, title="Alarm Type Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Top affected sites
    site_counts = sub_df[site_col].value_counts().reset_index()
    site_counts.columns = ['Site', 'Count']
    fig = px.bar(site_counts.head(10), x="Site", y="Count", color="Count", title="Top Sites for Severity")
    st.plotly_chart(fig, use_container_width=True)

# --- Navigation ---
st.markdown("---")
left, right = st.columns(2)
with left:
    st.markdown("[← Back to Visualizations](/pages/5_visualizations.py)")
with right:
    st.markdown("[Continue to Architecture →](/pages/7_architecture.py)")
