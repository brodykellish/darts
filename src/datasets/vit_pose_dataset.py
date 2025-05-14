import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ViTPoseDataset(Dataset):
    """Dataset for training the ViTPosePredictor model with enhanced augmentation."""

    def __init__(self, data_dir, image_size=224, augment=True):
        """
        Args:
            data_dir (str): Directory containing the dataset
            image_size (int): Size to resize images to (ViT models typically use 224x224)
            augment (bool): Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.example_dirs = sorted([d for d in os.listdir(data_dir)
                                  if os.path.isdir(os.path.join(data_dir, d))])
        self.image_size = image_size
        self.augment = augment

        # Base transforms that are always applied
        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])  # ViT normalization
        ])

        # Augmentation transforms that are only applied during training
        self.augmentation = transforms.Compose([
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=0),
        ])

    def __len__(self):
        return len(self.example_dirs)

    def __getitem__(self, idx):
        ex_dir = self.example_dirs[idx]
        ex_path = self.data_dir / ex_dir

        # Load image
        img_path = ex_path / 'image.png'
        image = Image.open(img_path).convert('RGB')

        # Apply base transform
        image = self.base_transform(image)

        # Apply augmentation if enabled
        if self.augment:
            image = self.augmentation(image)

        # Try to load quaternion data first, fall back to Euler angles if not available
        quat_path = ex_path / 'quaternion.txt'
        if os.path.exists(quat_path):
            # Load quaternion (w, x, y, z)
            rotation = torch.tensor(np.loadtxt(quat_path), dtype=torch.float32)

        # For now, we'll use a dummy translation (all zeros)
        # since our dataset only contains rotation information
        translation = torch.zeros(3, dtype=torch.float32)

        return {
            'image': image,
            'rotation': rotation,
            'translation': translation
        }
