import torch
import torch.nn as nn
import torch.nn.functional as F

class PosePredictor(nn.Module):
    """Model that predicts pose (rotation and translation) from images."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # CNN feature extractor
        self.encoder = nn.Sequential(
            # Input: 3 x 256 x 256
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 64 x 64 x 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 x 32 x 32
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 x 16 x 16
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 512 x 8 x 8
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Pose prediction heads
        self.rotation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D rotation (Euler angles)
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D translation
        )
        
    def forward(self, x):
        # Extract features from image
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        features = self.fc(features)
        
        # Predict pose
        rotation = self.rotation_head(features)
        translation = self.translation_head(features)
        
        return rotation, translation
    
    def encode(self, x):
        """Get only the feature representation."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc(features) 