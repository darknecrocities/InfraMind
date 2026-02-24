# 🏙️ InfraMind — AI That Understands Infrastructure Health From Visual Change

InfraMind is a high-performance, full-stack AI system designed to monitor infrastructure health through multi-temporal visual analysis. It detects structural damage, segments cracks, and predicts deterioration risk using advanced computer vision and machine learning.

## 🚀 Key Features
- **Multi-Temporal Image Ingestion**: Automated loading and normalization of drone, CCTV, and satellite imagery.
- **Image Alignment**: High-precision ORB-based alignment for accurate change detection.
- **Siamese Change Detection**: U-Net based Siamese architecture for identifying visual shifts over time.
- **Damage Segmentation**: Deep learning segmentation for precise structural damage mapping.
- **Risk Forecasting**: ML-driven regression (XGBoost) to predict failure timelines and risk levels.
- **Interactive Dashboard**: Real-time visualization using Streamlit and Plotly.

## 🛠️ Tech Stack
- **Vision**: PyTorch, OpenCV, Segmentation Models Pytorch (SMP)
- **ML**: Scikit-Learn, XGBoost
- **Data**: Pandas, Numpy, Albumentations
- **Dashboard**: Streamlit, Plotly

## 📁 Project Structure
```text
inframind/
│
├── data/               # Raw and processed data
├── src/                # Core AI and ML modules
├── dashboard/          # Streamlit application
├── checkpoints/        # Saved model weights
├── config/             # Configuration files
└── requirements.txt    # Project dependencies
```

## 🏗️ Installation & Setup
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
