import torch
import torch.nn as nn
from transformers import ViTModel

class ViTPosePredictor(nn.Module):
    def __init__(self, pretrained="google/vit-base-patch16-224"):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained)
        hidden_size = self.backbone.config.hidden_size
        
        # Pose prediction heads
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D rotation (Euler angles)
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D translation
        )
        
    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0]  # Use CLS token
        
        rotation = self.rotation_head(features)
        translation = self.translation_head(features)
        
        return rotation, translation