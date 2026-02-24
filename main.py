import os
import torch
import numpy as np
from src.ingestion.loader import TimeSeriesDataset
from src.preprocessing.alignment import align_images
from src.change_detection.model import SiameseUNet
from src.segmentation.model import DamageSegmenter
from src.feature_engineering.features import extract_features, save_features_to_csv
from src.regression.model import RiskPredictor
from src.utils.logging_utils import setup_logging
import yaml

logger = setup_logging()

def run_pipeline(data_root="data/raw"):
    """
    Run full inference pipeline from raw images to risk prediction.
    """
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    dataset = TimeSeriesDataset(data_root=data_root)
    segmenter = DamageSegmenter()
    predictor = RiskPredictor()
    
    all_features = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None: continue
        
        loc_id = sample['location_id']
        images = sample['images']
        timestamps = sample['timestamps']
        
        prev_mask = None
        for t in range(len(images)):
            img = images[t]
            ts = timestamps[t]
            
            logger.info(f"Processing {loc_id} at {ts}...")
            
            # 1. Damage Segmentation
            mask, area = segmenter.predict(img)
            
            # Convert img tensor to numpy for feature extraction
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            # 2. Change Detection (if t > 0)
            change_score = 0.0
            if t > 0:
                # In real scenario, load Siamese model and predict
                # For demo, we simulate a change score
                change_score = np.sum(np.abs(mask - prev_mask)) / mask.size if prev_mask is not None else 0.0
            
            # 3. Feature Extraction
            feat = extract_features(loc_id, ts, mask, image_np=img_np, prev_damage_mask=prev_mask, change_score=change_score)
            all_features.append(feat)
            
            prev_mask = mask

    # Save and predict
    save_features_to_csv(all_features)
    
    # Train/Update Risk Model
    predictor.train()
    
    logger.info("Pipeline Execution Complete.")

if __name__ == "__main__":
    run_pipeline()
