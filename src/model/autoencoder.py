import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    """Autoencoder that reconstructs images and predicts rotation from latent space."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Encoder
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
        
        # Latent space projection
        self.fc = nn.Sequential(
            nn.Linear(512, latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            # Start from 1x1xlatent_dim
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 32 x 64 x 64
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 16 x 128 x 128
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Rotation prediction head
        self.rotation_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3)  # 3D rotation (Euler angles)
        )
        
    def forward(self, x):
        # Encode
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        latent = self.fc(features)
        
        # Decode
        latent_reshaped = latent.view(latent.size(0), -1, 1, 1)
        output = self.decoder(latent_reshaped)
        
        return output, latent
    
    def encode(self, x):
        """Get only the latent representation."""
        features = self.encoder(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc(features) 