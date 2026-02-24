import numpy as np
import pandas as pd
import os
import torch
from src.utils.logging_utils import setup_logging

logger = setup_logging()

def extract_features(location_id, timestamp, damage_mask, prev_damage_mask=None, change_score=0.0):
    """
    Extract features from damage and change masks.
    """
    # Damage area
    damage_area_percent = (np.sum(damage_mask) / (damage_mask.shape[0] * damage_mask.shape[1])) * 100
    
    # Growth rate
    growth_rate = 0.0
    if prev_damage_mask is not None:
        prev_area = (np.sum(prev_damage_mask) / (prev_damage_mask.shape[0] * prev_damage_mask.shape[1])) * 100
        if prev_area > 0:
            growth_rate = (damage_area_percent - prev_area) / prev_area
        else:
            growth_rate = damage_area_percent # New damage
            
    # Mock texture variance and color degradation for simulation
    texture_variance = np.var(damage_mask) # Simplified
    color_degradation = np.mean(damage_mask) * 0.5 # Simplified
    
    features = {
        "location_id": location_id,
        "date": timestamp,
        "crack_area_percent": float(damage_area_percent),
        "growth_rate": float(growth_rate),
        "texture_variance": float(texture_variance),
        "color_degradation": float(color_degradation),
        "change_score": float(change_score)
    }
    
    return features

def save_features_to_csv(features_list, output_path="data/features/infrastructure_features.csv"):
    """
    Save list of feature dictionaries to CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(features_list)
    
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)
        
    logger.info(f"Saved {len(features_list)} feature records to {output_path}")

if __name__ == "__main__":
    # Test feature extraction
    mask1 = np.zeros((512, 512))
    mask2 = np.ones((512, 512)) * 0.1 # Simulate some damage
    
    feat = extract_features("LOC_001", "2023-01-01", mask2, mask1)
    print(feat)
    save_features_to_csv([feat])
