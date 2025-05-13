import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

class PoseDataset(Dataset):
    """Dataset for loading rendered images and their corresponding poses."""
    
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory with rendered images and pose files
            transform (callable, optional): Optional transform to be applied on images
        """
        self.data_dir = data_dir
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.png')])
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)
        
        # Load pose
        pose_name = os.path.join(self.data_dir, self.image_files[idx].replace('.png', '.txt'))
        pose = np.loadtxt(pose_name)
        R = pose[:, :3]  # Rotation matrix
        t = pose[:, 3]   # Translation vector
        
        # Convert rotation matrix to Euler angles
        euler = torch.tensor([
            np.arctan2(R[2, 1], R[2, 2]),
            np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2)),
            np.arctan2(R[1, 0], R[0, 0])
        ], dtype=torch.float32)
        
        # Convert translation to tensor
        translation = torch.tensor(t, dtype=torch.float32)
        
        return {
            'image': image,
            'rotation': euler,
            'translation': translation
        } 