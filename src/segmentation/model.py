import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import numpy as np
from src.utils.logging_utils import setup_logging

logger = setup_logging()

class DamageSegmenter:
    """
    Damage segmentation model using SMP U-Net.
    """
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation='sigmoid'
        ).to(self.device)
        self.model.eval()

    def predict(self, image_tensor):
        """
        Predict damage mask for a single image tensor.
        Returns: mask (numpy), damage_area_percent (float)
        """
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        with torch.no_grad():
            mask = self.model(image_tensor.to(self.device))
            mask = (mask > 0.5).float().cpu().numpy().squeeze()
            
        damage_area_percent = (np.sum(mask) / (mask.shape[0] * mask.shape[1])) * 100
        return mask, damage_area_percent

if __name__ == "__main__":
    segmenter = DamageSegmenter()
    dummy_img = torch.randn(1, 3, 512, 512)
    mask, area = segmenter.predict(dummy_img)
    print(f"Mask shape: {mask.shape}, Area: {area:.2f}%")
