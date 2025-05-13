import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicAutoEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, 4, 2, 1)  # (B, 32, 64, 64)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1) # (B, 64, 32, 32)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1) # (B, 128, 16, 16)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1) # (B, 256, 8, 8)
        self.enc_fc = nn.Linear(256*8*8, latent_dim)

        # Decoder
        self.dec_fc = nn.Linear(latent_dim, 256*8*8)
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_deconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_deconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_deconv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)

    def encoder(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)
        x = self.enc_fc(x)
        return x

    def decoder(self, z):
        x = self.dec_fc(z)
        x = x.view(x.size(0), 256, 8, 8)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = F.relu(self.dec_deconv3(x))
        x = torch.sigmoid(self.dec_deconv4(x))
        return x

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z 