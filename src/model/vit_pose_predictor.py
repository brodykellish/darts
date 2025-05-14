import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

class ViTPosePredictor(nn.Module):
    def __init__(self, pretrained="google/vit-base-patch16-224", dropout_rate=0.3, freeze_backbone_layers=8):
        super().__init__()
        self.backbone = ViTModel.from_pretrained(pretrained)
        hidden_size = self.backbone.config.hidden_size

        # Optionally freeze some layers of the backbone to prevent overfitting
        if freeze_backbone_layers > 0:
            # Freeze the embedding layer and first N encoder layers
            for param in self.backbone.embeddings.parameters():
                param.requires_grad = False

            for i in range(min(freeze_backbone_layers, len(self.backbone.encoder.layer))):
                for param in self.backbone.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Improved rotation head with dropout and batch normalization
        self.rotation_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3)  # 3D rotation (Euler angles)
        )

        # Improved translation head with dropout and batch normalization
        self.translation_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 3)  # 3D translation
        )

    def forward(self, x):
        outputs = self.backbone(x)
        features = outputs.last_hidden_state[:, 0]  # Use CLS token

        rotation = self.rotation_head(features)
        translation = self.translation_head(features)

        return rotation, translation

    def get_quaternion(self, euler_angles):
        """Convert Euler angles to quaternion for better rotation representation"""
        # This is a simplified conversion - in practice you might want to use a library
        # like scipy.spatial.transform.Rotation for more robust conversion
        x, y, z = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

        # Convert to quaternion (simplified implementation)
        qx = torch.sin(x/2) * torch.cos(y/2) * torch.cos(z/2) - torch.cos(x/2) * torch.sin(y/2) * torch.sin(z/2)
        qy = torch.cos(x/2) * torch.sin(y/2) * torch.cos(z/2) + torch.sin(x/2) * torch.cos(y/2) * torch.sin(z/2)
        qz = torch.cos(x/2) * torch.cos(y/2) * torch.sin(z/2) - torch.sin(x/2) * torch.sin(y/2) * torch.cos(z/2)
        qw = torch.cos(x/2) * torch.cos(y/2) * torch.cos(z/2) + torch.sin(x/2) * torch.sin(y/2) * torch.sin(z/2)

        return torch.stack([qw, qx, qy, qz], dim=1)