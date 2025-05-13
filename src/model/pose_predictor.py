import torch
import torch.nn as nn
from .autoencoder import Autoencoder

class PosePredictor(nn.Module):
    """Model that predicts pose (rotation and translation) from latent space."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        self.autoencoder = Autoencoder(latent_dim)
        
        # Pose prediction layers
        self.rotation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3D rotation (Euler angles)
        )
        
        self.translation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3D translation
        )
        
    def forward(self, x):
        # Get latent representation
        _, latent = self.autoencoder(x)
        
        # Predict pose
        rotation = self.rotation_head(latent)
        translation = self.translation_head(latent)
        
        return rotation, translation
    
    def encode(self, x):
        """Get only the latent representation."""
        _, latent = self.autoencoder(x)
        return latent
    
    def decode(self, latent):
        """Decode from latent space."""
        return self.autoencoder.decoder(latent) 