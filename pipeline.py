import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from lib.networks import AutoEncoder  # from AAE repo

# === CONFIG ===
MODEL_PATH = "checkpoints/dart_model.pth"
CODEBOOK_LATENTS_PATH = "codebook_latents.npy"
CODEBOOK_POSES_PATH = "codebook_poses.npy"
IMAGE_PATH = "sample_image.jpg"  # Full RGB image with dart
CROP_BBOX = (100, 150, 200, 250)  # Replace with dart detection box (x1, y1, x2, y2)

IMAGE_RES = 128
LATENT_DIM = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Known 3D coordinates of dart tip in model space
TIP_MODEL = np.array([0.0, 0.0, 0.05])  # Adjust to your model

# Camera intrinsics (K) — replace with your calibrated values
K = np.array([
    [600, 0, 320],
    [0, 600, 240],
    [0,   0,   1]
])

# === Load components ===
model = AutoEncoder.load_from_checkpoint(MODEL_PATH)
model.eval().to(device)

codebook_latents = np.load(CODEBOOK_LATENTS_PATH)
codebook_poses = np.load(CODEBOOK_POSES_PATH)

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((IMAGE_RES, IMAGE_RES)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# === Load full image ===
img_full = Image.open(IMAGE_PATH).convert("RGB")
img_cv = cv2.imread(IMAGE_PATH)  # For visualization

# Crop dart region (replace with auto-crop later)
x1, y1, x2, y2 = CROP_BBOX
dart_crop = img_full.crop((x1, y1, x2, y2))
dart_tensor = transform(dart_crop).unsqueeze(0).to(device)

# === Encode + Nearest Neighbor Match ===
with torch.no_grad():
    latent = model.encoder(dart_tensor).cpu().numpy().squeeze()

# Find NN in latent space
dists = np.linalg.norm(codebook_latents - latent, axis=1)
nn_idx = np.argmin(dists)
pose = codebook_poses[nn_idx]  # shape (3, 4) → [R | t]

R = pose[:, :3]
t = pose[:, 3]

# === Project dart tip to image ===
tip_cam = R @ TIP_MODEL + t  # In camera coords
tip_img_h = K @ tip_cam      # Homogeneous projection
tip_img = tip_img_h[:2] / tip_img_h[2]  # Normalize to (x, y)

print(f"Dart tip pixel location: x={tip_img[0]:.1f}, y={tip_img[1]:.1f}")

# === Draw tip on original image ===
tip_x, tip_y = int(tip_img[0]), int(tip_img[1])
cv2.circle(img_cv, (tip_x, tip_y), 5, (0, 255, 0), -1)
cv2.imwrite("dart_tip_predicted.jpg", img_cv)

