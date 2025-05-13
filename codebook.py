import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from lib.networks import AutoEncoder  # from AugmentedAutoencoder repo

RENDER_DIR = "/absolute/path/to/dart_dataset/renders"
MODEL_PATH = "checkpoints/dart_model.pth"
OUT_CODEBOOK = "codebook_latents.npy"
OUT_POSES = "codebook_poses.npy"

IMAGE_RES = 128  # Should match your AAE training resolution
LATENT_DIM = 128  # Should match your AAE architecture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Image preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_RES, IMAGE_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Match AAE training norm
])

# 2. Load model
model = AutoEncoder.load_from_checkpoint(MODEL_PATH)
model.eval().to(device)

# 3. Gather image paths
img_files = sorted([f for f in os.listdir(RENDER_DIR) if f.endswith(".png")])
latent_codes = []
poses = []

for fname in tqdm(img_files, desc="Generating codebook"):
    img_path = os.path.join(RENDER_DIR, fname)
    pose_path = img_path.replace(".png", ".txt")

    # Load and preprocess image
    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        latent = model.encoder(tensor).cpu().numpy().squeeze()

    latent_codes.append(latent)

    # Load associated pose
    pose = np.loadtxt(pose_path).reshape(3, 4)
    poses.append(pose)

# 4. Save codebook
latent_codes = np.array(latent_codes)  # shape [N, latent_dim]
poses = np.array(poses)                # shape [N, 3, 4]

np.save(OUT_CODEBOOK, latent_codes)
np.save(OUT_POSES, poses)

print(f"Saved codebook with {len(latent_codes)} entries.")

