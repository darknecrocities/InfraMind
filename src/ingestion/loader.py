import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class TimeSeriesDataset(Dataset):
    """
    Dataset class for multi-temporal infrastructure images.
    Expects a folder structure where each subdirectory is a 'location_id'
    containing images named by timestamp (e.g., location1/2023-01-01.jpg, location1/2023-06-01.jpg).
    """
    def __init__(self, data_root, img_size=(512, 512), transform=None):
        self.data_root = data_root
        self.img_size = img_size
        self.transform = transform or T.Compose([
            T.Resize(self.img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.locations = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        logger.info(f"Initialized TimeSeriesDataset with {len(self.locations)} locations.")

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, idx):
        location_id = self.locations[idx]
        location_path = os.path.join(self.data_root, location_id)
        
        # Get all image files in the location directory
        img_files = sorted([f for f in os.listdir(location_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if len(img_files) < 2:
            logger.warning(f"Location {location_id} has fewer than 2 images. Skipping.")
            return None

        images = []
        timestamps = []
        
        for img_file in img_files:
            img_path = os.path.join(location_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                tensor_img = self.transform(img)
                images.append(tensor_img)
                timestamps.append(img_file.split('.')[0]) # Use filename as timestamp
            except Exception as e:
                logger.error(f"Error loading image {img_path}: {e}")
                continue

        return {
            "location_id": location_id,
            "timestamps": timestamps,
            "images": images
        }

if __name__ == "__main__":
    # Test script
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    test_root = config['paths']['raw_data']
    # Create test directory if it doesn't exist
    os.makedirs(test_root, exist_ok=True)
    
    dataset = TimeSeriesDataset(data_root=test_root)
    print(f"Dataset size: {len(dataset)}")
