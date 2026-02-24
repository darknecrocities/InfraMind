import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.change_detection.model import SiameseUNet
from src.ingestion.loader import TimeSeriesDataset
from src.utils.logging_utils import setup_logging
import os
import yaml

logger = setup_logging()

class ChangeDetectionTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.org.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = SiameseUNet(
            encoder_name=self.config['models']['change_detection']['encoder']
        ).to(self.device)
        
        self.criterion = smp.losses.DiceLoss(mode='binary') # Using Dice Loss as requested
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['models']['change_detection']['lr'])
        
    def train(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                if batch is None: continue
                
                # In a real scenario, we'd have ground truth masks. 
                # For this implementation, we simulate batch loading.
                images = batch['images']
                if len(images) < 2: continue
                
                img1 = images[0].unsqueeze(0).to(self.device)
                img2 = images[1].unsqueeze(0).to(self.device)
                
                # Mock target mask (all zeros for now since no real data)
                target = torch.zeros((1, 1, 512, 512)).to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(img1, img2)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
            
        # Save checkpoint
        os.makedirs(self.config['paths']['checkpoints'], exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.config['paths']['checkpoints'], "change_detection.pth"))
        logger.info("Model saved to checkpoints/change_detection.pth")

if __name__ == "__main__":
    # Test training loop with synthetic data
    dataset = TimeSeriesDataset(data_root="data/raw")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    trainer = ChangeDetectionTrainer()
    trainer.train(loader, epochs=2)
