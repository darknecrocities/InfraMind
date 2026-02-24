import numpy as np
import pandas as pd
import os
import torch
import cv2
from src.utils.logging_utils import setup_logging

logger = setup_logging()

def extract_features(location_id, timestamp, damage_mask, image_np=None, prev_damage_mask=None, change_score=0.0):
    """
    Extract features from damage and change masks.
    Included damage severity based on image context if available.
    """
    # Damage area
    total_pixels = damage_mask.shape[0] * damage_mask.shape[1]
    damage_area_percent = (np.sum(damage_mask) / total_pixels) * 100
    
    # Damage Severity (Intensity of damage relative to background)
    severity_score = 0.0
    max_severity = 0.0
    complexity = 0.0
    if image_np is not None and np.sum(damage_mask) > 0:
        # Match image shape to mask shape for correct indexing
        if image_np.shape[:2] != damage_mask.shape:
            image_np = cv2.resize(image_np, (damage_mask.shape[1], damage_mask.shape[0]))
            
        # Normalize image to 0-1
        img_gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        img_norm = img_gray / 255.0
        
        # Calculate mean intensity of damaged vs non-damaged area
        damage_pixels = img_norm[damage_mask > 0]
        damage_intensity = np.mean(damage_pixels)
        # A simple heuristic: darker = deeper crack (for concrete)
        severity_score = (1.0 - damage_intensity) * 10
        max_severity = (1.0 - np.min(damage_pixels)) * 10
        
        # Complexity (Perimeter-to-Area ratio)
        mask_uint8 = damage_mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = 0
        for cnt in contours:
            perimeter += cv2.arcLength(cnt, True)
        
        area = np.sum(damage_mask)
        if area > 0:
            complexity = perimeter / np.sqrt(area) # Scale-invariant complexity
        
    # Growth rate
    growth_rate = 0.0
    if prev_damage_mask is not None:
        prev_area = (np.sum(prev_damage_mask) / total_pixels) * 100
        if prev_area > 0:
            growth_rate = (damage_area_percent - prev_area) / prev_area
        else:
            growth_rate = damage_area_percent # New damage
            
    # Texture variance in damaged area
    texture_variance = np.var(damage_mask) 
    color_degradation = severity_score * 0.1
    
    features = {
        "location_id": location_id,
        "date": timestamp,
        "crack_area_percent": float(damage_area_percent),
        "severity_score": float(severity_score),
        "max_severity": float(max_severity),
        "complexity": float(complexity),
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
