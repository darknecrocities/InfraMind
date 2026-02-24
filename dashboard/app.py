import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import torch
import cv2

# Set Page Config
st.set_page_config(page_title="🏙️ InfraMind — AI Infrastructure Monitor", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ InfraMind — AI Infrastructure Monitoring")
st.markdown("---")

# Sidebar
st.sidebar.header("Settings")
data_path = "data/features/infrastructure_features.csv"
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    locations = df['location_id'].unique()
    selected_loc = st.sidebar.selectbox("Select Infrastructure Location", locations)
    
    loc_df = df[df['location_id'] == selected_loc].sort_values('date')
else:
    st.sidebar.error("No feature data found. Please run the analysis pipeline first.")
    st.stop()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Visual Analysis")
    # Time-lapse slider
    timestep = st.slider("Select Timestep", 0, len(loc_df)-1, 0)
    selected_row = loc_df.iloc[timestep]
    
    # Load dummy images for display (In real app, load from data/raw)
    st.write(f"Displaying data for Date: {selected_row['date']}")
    
    # Visual placeholders
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        st.info("Original Frame")
        st.image("https://via.placeholder.com/512x512.png?text=Infrastructure+Image", use_container_width=True)
    with img_col2:
        st.info("Damage Heatmap")
        st.image("https://via.placeholder.com/512x512.png?text=Segmentation+Mask", use_container_width=True)

with col2:
    st.subheader("Risk Metrics")
    
    # Risk Score Simulation (if not in CSV)
    risk_score = selected_row.get('risk_score', selected_row['crack_area_percent'] * 10)
    risk_level = "LOW"
    if risk_score > 70: risk_level = "HIGH"
    elif risk_score > 30: risk_level = "MEDIUM"
    
    # Gauge Chart
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risk Score (0-100)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "#1f77b4"},
            'steps' : [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}]
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Damage Area", f"{selected_row['crack_area_percent']:.2f}%", f"{selected_row['growth_rate']:.2f}%")
    m_col2.metric("Risk Level", risk_level, delta_color="inverse")

# Trend Analysis
st.markdown("---")
st.subheader("Historical Trends")
fig_trend = px.line(loc_df, x='date', y='crack_area_percent', title='Damage Progression Over Time', markers=True)
fig_trend.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig_trend, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.write("Developed by Antigravity AI")
