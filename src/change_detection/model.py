import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SiameseUNet(nn.Module):
    """
    Siamese U-Net for change detection.
    Uses a shared encoder to extract features from two images,
    then concatenates or subtracts features to predict a change mask.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super(SiameseUNet, self).__init__()
        
        # We use SMP for the base U-Net architecture
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1, # Binary change mask
            activation=None # We'll apply sigmoid in loss/inference
        )
        
        # The encoder is shared
        self.encoder = self.base_model.encoder
        self.decoder = self.base_model.decoder
        self.segmentation_head = self.base_model.segmentation_head

    def forward(self, x1, x2):
        """
        x1: Image at t1 (B, 3, H, W)
        x2: Image at t2 (B, 3, H, W)
        """
        # Feature extraction (shared encoder)
        feat1 = self.encoder(x1)
        feat2 = self.encoder(x2)
        
        # Fusion: Difference of features
        fused_features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]
        
        # Decode to mask
        decoder_output = self.decoder(*fused_features)
        mask = self.segmentation_head(decoder_output)
        
        return mask

if __name__ == "__main__":
    # Test model
    model = SiameseUNet()
    x1 = torch.randn(1, 3, 512, 512)
    x2 = torch.randn(1, 3, 512, 512)
    out = model(x1, x2)
    print(f"Output mask shape: {out.shape}")
