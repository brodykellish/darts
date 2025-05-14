#!/usr/bin/env python3
"""
Script to visualize predictions from the trained ViTPosePredictor model.
This script:
1. Loads a trained ViTPosePredictor model
2. Makes predictions on a sample from the dataset
3. Visualizes the true and predicted rotations
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation
from torchvision import transforms

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.vit_pose_predictor import ViTPosePredictor
from src.render.stl_renderer import load_and_normalize_stl, render_mesh_with_vectors


def load_model(model_path):
    """
    Load a trained ViTPosePredictor model.

    Args:
        model_path (str): Path to the model checkpoint

    Returns:
        model: Loaded model
    """
    # Create model
    model = ViTPosePredictor()

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    return model


def load_sample(data_dir, sample_idx):
    """
    Load a sample from the dataset.

    Args:
        data_dir (str): Directory containing the dataset
        sample_idx (int): Index of the sample to load
        use_quaternion (bool): Whether to use quaternion representation

    Returns:
        image: PIL image
        rotation: True rotation as numpy array (Euler angles or quaternion)
        is_quaternion: Whether the returned rotation is a quaternion
    """
    # Get sample directory
    sample_dir = os.path.join(data_dir, f"{sample_idx:04d}")

    # Load image
    img_path = os.path.join(sample_dir, "image.png")
    image = Image.open(img_path).convert('RGB')

    # Try to load quaternion
    quat_path = os.path.join(sample_dir, "quaternion.txt")
    rotation = np.loadtxt(quat_path)

    return image, rotation



def predict_rotation(model, image, image_size=224):
    """
    Predict rotation for an image.

    Args:
        model: ViTPosePredictor model
        image: PIL image
        image_size (int): Size to resize image to

    Returns:
        rotation: Predicted rotation as numpy array (quaternion)
    """
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Transform image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        pred_rot, _ = model(img_tensor)

    # Convert to numpy
    pred_rot = pred_rot.squeeze().numpy()

    return pred_rot


def visualize_rotations(stl_mesh, true_rot, pred_rot=None):
    """
    Visualize true and predicted rotations.

    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        true_rot: True rotation as Euler angles (xyz) or quaternion (w, x, y, z)
        pred_rot: Predicted rotation as Euler angles (xyz) or quaternion (w, x, y, z)
        is_quaternion: Whether the rotations are quaternions
        figsize: Figure size
    """
    # Convert to rotation objects
    # For quaternions in (w, x, y, z) format, convert to scipy's (x, y, z, w) format
    true_quat = np.array([true_rot[1], true_rot[2], true_rot[3], true_rot[0]])
    true_rotation = Rotation.from_quat(true_quat)

    if pred_rot is not None:
        pred_quat = np.array([pred_rot[1], pred_rot[2], pred_rot[3], pred_rot[0]])
        pred_rotation = Rotation.from_quat(pred_quat)

    render_mesh_with_vectors(
        stl_mesh=stl_mesh,
        rotations=[true_rotation, pred_rotation],
        show=True,
        show_vec=True
    )


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize ViTPosePredictor predictions')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--sample_idx', type=int, default=0, help='Index of the sample to visualize')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the visualization')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)

    # Load sample
    print(f"Loading sample {args.sample_idx} from {args.data_dir}...")
    image, true_rot = load_sample(args.data_dir, args.sample_idx)

    # Predict rotation
    print("Predicting rotation...")
    pred_rot = predict_rotation(model, image)

    # Print rotations
    print(f"True rotation (quaternion, wxyz): {true_rot}")
    print(f"Predicted rotation (quaternion, wxyz): {pred_rot}")

    # Calculate quaternion distance (dot product)
    # Ensure unit quaternions
    true_quat_norm = true_rot / np.linalg.norm(true_rot)
    pred_quat_norm = pred_rot / np.linalg.norm(pred_rot)

    # The absolute dot product gives the cosine of half the rotation angle between the orientations
    dot_product = np.abs(np.dot(true_quat_norm, pred_quat_norm))
    dot_product = min(dot_product, 1.0)  # Clamp to avoid numerical issues

    # Calculate the angle between the quaternions
    angle_error = 2 * np.arccos(dot_product)
    print(f"Quaternion angle error: {angle_error:.4f} radians ({np.degrees(angle_error):.2f} degrees)")

    # Load and normalize STL
    print(f"Loading STL from {args.stl_path}...")
    stl_mesh = load_and_normalize_stl(args.stl_path)

    # Visualize
    print("Generating visualization...")
    fig = visualize_rotations(stl_mesh, true_rot, pred_rot)

    # Save or show
    if args.output_path:
        fig.savefig(args.output_path)
        print(f"Visualization saved to {args.output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
