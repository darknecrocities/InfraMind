import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import os
import torch
import cv2
import torchvision.transforms as T
from src.segmentation.model import DamageSegmenter
from src.regression.model import RiskPredictor

# Set Page Config
st.set_page_config(page_title="🏙️ InfraMind — AI Infrastructure Monitor", layout="wide")

# Cached Model Loading
@st.cache_resource
def load_models():
    segmenter = DamageSegmenter()
    predictor = RiskPredictor()
    # Try to load trained risk model
    if os.path.exists(predictor.model_path):
        import joblib
        predictor.model = joblib.load(predictor.model_path)
    return segmenter, predictor

segmenter, predictor = load_models()

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stMetric { background-color: #1e2227; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; font-weight: bold; }
    .risk-low { color: #00ff00; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏙️ InfraMind — AI Infrastructure Monitoring")
st.markdown("---")

tab1, tab2 = st.tabs(["📊 Historical Analysis", "🔍 Real-time Inspection"])

with tab1:
    # Sidebar for Historical
    st.sidebar.header("Historical Settings")
    data_path = "data/features/infrastructure_features.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        locations = df['location_id'].unique()
        selected_loc = st.sidebar.selectbox("Select Infrastructure Location", locations)
        loc_df = df[df['location_id'] == selected_loc].sort_values('date')
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Visual Analysis")
            timestep = st.slider("Select Timestep", 0, len(loc_df)-1, 0, key="hist_slider")
            selected_row = loc_df.iloc[timestep]
            st.write(f"Displaying data for Date: {selected_row['date']}")
            
            loc_dir = os.path.join("data/raw", selected_loc)
            image_file = None
            if os.path.exists(loc_dir):
                files = [f for f in os.listdir(loc_dir) if f.startswith(str(selected_row['date']))]
                if files: image_file = os.path.join(loc_dir, files[0])

            img_col1, img_col2 = st.columns(2)
            with img_col1:
                st.info("Original Frame")
                if image_file: st.image(image_file, use_container_width=True)
                else: st.image("https://via.placeholder.com/512x512.png?text=Image+Not+Found")
            with img_col2:
                st.info("Damage Heatmap (Simulated)")
                st.image("https://via.placeholder.com/512x512.png?text=Segmentation+Mask")

        with col2:
            st.subheader("Risk Metrics")
            risk_score = selected_row.get('risk_score', float(selected_row['crack_area_percent'] * 10))
            risk_level = "LOW"
            if risk_score > 70: risk_level = "HIGH"
            elif risk_score > 30: risk_level = "MEDIUM"
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = risk_score,
                gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#1f77b4"},
                        'steps' : [{'range': [0, 30], 'color': "green"}, {'range': [30, 70], 'color': "yellow"}, {'range': [70, 100], 'color': "red"}]}
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Damage Area", f"{selected_row['crack_area_percent']:.2f}%", f"{selected_row['growth_rate']:.2f}%")
            m_col2.metric("Risk Level", risk_level)

        st.markdown("---")
        st.subheader("Historical Trends")
        fig_trend = px.line(loc_df, x='date', y='crack_area_percent', title='Damage Progression Over Time', markers=True)
        fig_trend.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.warning("Please run the analysis pipeline first to see historical data.")

with tab2:
    st.subheader("Upload Infrastructure Image for Immediate Analysis")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        try:
            # 1. Load Image
            image = Image.open(uploaded_file).convert("RGB")
            st.success("Image uploaded successfully!")
            
            # 2. Preprocess
            transform = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            img_tensor = transform(image)
            
            with st.spinner("Analyzing structural integrity..."):
                # 3. Predict Segmentation
                mask, area_percent = segmenter.predict(img_tensor)
                
                # Convert PIL image to numpy for severity scoring
                image_np = np.array(image)
                
                # 4. Predict Risk using the same extraction logic as the pipeline
                from src.feature_engineering.features import extract_features
                features = extract_features(
                    "UPLOAD", "NOW", mask, 
                    image_np=image_np, 
                    change_score=0.0
                )
                risk_res = predictor.predict(features)
            
            # 5. Display Results
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.info("Analysis View")
                # Create overlay or side-by-side
                tab_img_orig, tab_img_mask = st.tabs(["Original", "Detected Damage"])
                with tab_img_orig:
                    st.image(image, use_container_width=True)
                with tab_img_mask:
                    # Create a heatmap look: Damage in Red/Orange
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    heatmap = cv2.applyColorMap(mask_uint8, cv2.COLORMAP_JET)
                    # Convert to RGB for Streamlit
                    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Overlay on original image for high-impact visual
                    image_resized = image.resize((512, 512))
                    image_np_resized = np.array(image_resized)
                    overlay = cv2.addWeighted(image_np_resized, 0.6, heatmap_rgb, 0.4, 0)
                    
                    st.image(overlay, caption="High-Impact Damage Localization Map", use_container_width=True)
            
            with res_col2:
                st.info("Risk Assessment")
                st.metric("Risk Score", f"{risk_res['risk_score']:.1f}/100")
                
                # Risk Gauge
                fig_rt_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number", value = risk_res['risk_score'],
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#1f77b4"},
                            'steps' : [{'range': [0, 30], 'color': "green"}, {'range': [30, 70], 'color': "yellow"}, {'range': [70, 100], 'color': "red"}]}
                ))
                fig_rt_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig_rt_gauge, use_container_width=True)
                
                st.markdown(f"**Structural Risk Level:** <span class='risk-{risk_res['risk_level'].lower()}'>{risk_res['risk_level']}</span>", unsafe_allow_html=True)
                st.write(f"**Damage Detected:** {area_percent:.2f}% of surface area.")
                st.write(f"**Recommendation:** " + 
                         ("IMMEDIATE MAINTENANCE REQUIRED" if risk_res['risk_level'] == "HIGH" else 
                          "SCHEDULE INSPECTION WITHIN 3 MONTHS" if risk_res['risk_level'] == "MEDIUM" else 
                          "ROUTINE MONITORING"))

        except Exception as e:
            st.error(f"Error processing image: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Developed by Antigravity AI")
