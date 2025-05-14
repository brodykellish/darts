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
        
    Returns:
        image: PIL image
        rotation: True rotation as numpy array
    """
    # Get sample directory
    sample_dir = os.path.join(data_dir, f"{sample_idx:04d}")
    
    # Load image
    img_path = os.path.join(sample_dir, "image.png")
    image = Image.open(img_path).convert('RGB')
    
    # Load rotation
    rot_path = os.path.join(sample_dir, "rotation.txt")
    rotation = np.loadtxt(rot_path)
    
    return image, rotation


def predict_rotation(model, image, image_size=224):
    """
    Predict rotation for an image.
    
    Args:
        model: ViTPosePredictor model
        image: PIL image
        image_size (int): Size to resize image to
        
    Returns:
        rotation: Predicted rotation as numpy array
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


def visualize_rotations(vertices, faces, true_rot, pred_rot=None, figsize=(10, 5)):
    """
    Visualize true and predicted rotations.
    
    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        true_rot: True rotation as Euler angles (xyz)
        pred_rot: Predicted rotation as Euler angles (xyz)
        figsize: Figure size
    """
    # Convert Euler angles to rotation objects
    true_rotation = Rotation.from_euler('xyz', true_rot)
    
    if pred_rot is not None:
        pred_rotation = Rotation.from_euler('xyz', pred_rot)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': '3d'})
        axes = [ax1, ax2]
        titles = ['True Rotation', 'Predicted Rotation']
        rotations = [true_rotation, pred_rotation]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]), subplot_kw={'projection': '3d'})
        axes = [ax]
        titles = ['True Rotation']
        rotations = [true_rotation]
    
    # Plot each rotation
    for ax, title, rotation in zip(axes, titles, rotations):
        # Apply rotation to vertices
        rotated_vertices = rotation.apply(vertices)
        
        # Plot the mesh
        ax.plot_trisurf(
            rotated_vertices[:, 0],
            rotated_vertices[:, 1],
            rotated_vertices[:, 2],
            triangles=faces,
            color='lightgray',
            edgecolor=None,
            linewidth=0.2,
            alpha=0.8
        )
        
        # Add rotation vector (unit vector along z-axis)
        origin = np.zeros(3)
        z_axis = np.array([0, 0, 1])
        rotated_z = rotation.apply(z_axis)
        
        # Plot rotation vector
        ax.quiver(
            origin[0], origin[1], origin[2],
            rotated_z[0], rotated_z[1], rotated_z[2],
            color='red', linewidth=2, arrow_length_ratio=0.15
        )
        
        # Set equal aspect ratio and remove axes
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_title(title)
    
    plt.tight_layout()
    return fig


def load_and_normalize_stl(file_path):
    """
    Load an STL file and normalize it to be centered at the origin with unit scale.
    
    Args:
        file_path: Path to the STL file
        
    Returns:
        vertices and faces arrays
    """
    from stl import mesh
    
    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)
    
    # Extract vertices and faces
    triangles = stl_mesh.vectors
    vertices_all = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(vertices_all.round(decimals=5), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    
    # Center at origin
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    
    # Scale to unit size
    scale = np.max(np.abs(vertices))
    vertices = vertices / scale
    
    return vertices, faces


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
    print(f"True rotation (Euler angles, xyz): {true_rot}")
    print(f"Predicted rotation (Euler angles, xyz): {pred_rot}")
    
    # Calculate error
    error = np.abs(true_rot - pred_rot)
    mean_error = np.mean(error)
    print(f"Mean absolute error: {mean_error:.4f} radians ({np.degrees(mean_error):.2f} degrees)")
    
    # Load and normalize STL
    print(f"Loading STL from {args.stl_path}...")
    vertices, faces = load_and_normalize_stl(args.stl_path)
    
    # Visualize
    print("Generating visualization...")
    fig = visualize_rotations(vertices, faces, true_rot, pred_rot)
    
    # Save or show
    if args.output_path:
        fig.savefig(args.output_path)
        print(f"Visualization saved to {args.output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
