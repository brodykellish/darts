#!/usr/bin/env python3
"""
Data generation script that creates a diverse dataset of rotated/lighted renders of a 3D model.

This script:
- Loads a specified .stl file, normalized to be centered at the origin with unit scale
- Accepts parameters for image size, number of samples, output directory, etc.
- Generates N samples with random rotations using various sampling methods
- Adds more variation in lighting, viewpoints, and backgrounds
- Saves renders and rotation information (Euler angles and quaternions) in the specified output directory
- Can visualize the generated dataset with side-by-side comparisons of original and reconstructed images
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from scipy.spatial.transform import Rotation
from stl import mesh

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.geometry.rotation_utils import generate_uniform_rotations
from src.render.stl_renderer import render_mesh, load_and_normalize_stl, render_mesh_with_vectors

def generate_dataset(stl_mesh, num_samples, output_dir, image_size=256,
                     rotation_method='fibonacci', visualize=False):
    """
    Generate a dataset of rendered images with diverse rotations.

    Args:
        vertices: Numpy array of vertices
        faces: Numpy array of faces
        num_samples: Number of samples to generate
        output_dir: Directory to save the samples
        image_size: Size of the rendered image (square)
        rotation_method: Method for generating rotations
        visualize: Whether to visualize the first sample to allow user to cancel if it looks wrong
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate rotations
    print(f"Generating {num_samples} rotations using {rotation_method} method...")
    rotations = generate_uniform_rotations(num_samples, method=rotation_method)

    for i in tqdm(range(num_samples)):
        if i == 1 and visualize:  # Visualize the first sample to allow user to cancel if it looks wrong
            print("Visualizing first sample to allow user to cancel if it looks wrong...")
            visualize_sample(
                stl_mesh=stl_mesh,
                data_dir=output_dir,
                sample_idx=0,
                image_size=image_size,
            )
            should_continue = input("Continue generating dataset? (y/n): ")
            if should_continue.lower() != 'y':
                print("User cancelled dataset generation.")
                return

        # Create subdirectory for this sample
        sample_dir = os.path.join(output_dir, f"{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Get rotation
        rotation = rotations[i]

        # Get quaternion (x, y, z, w)
        quaternion = rotation.as_quat()

        # Reorder to (w, x, y, z) which is more common
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])

        # Render image
        image = render_mesh(
            stl_mesh, rotation, image_size
        )

        # Save image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        Image.fromarray(image).save(os.path.join(sample_dir, "image.png"))

        # Save rotation data as quaternions
        np.savetxt(os.path.join(sample_dir, "quaternion.txt"), quaternion)


def visualize_sample(stl_mesh, data_dir, sample_idx, image_size=256):
    """
    Create an animation that visualizes the dataset by showing side-by-side comparisons
    of the original rendered images and reconstructions from the saved rotation data.

    Args:
        stl_mesh: The STL mesh
        data_dir: Directory containing the dataset
        sample_idx: Index of the sample to visualize
        image_size: Size of the rendered images
    """
    # Get all sample directories
    sample_dir = sorted([d for d in os.listdir(data_dir)
                        if os.path.isdir(os.path.join(data_dir, d))])
    sample_dir = sample_dir[sample_idx]

    # Create figure for animation
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    # Left subplot for original image
    ax1 = plt.subplot(gs[0])
    ax1.set_title("Original Image")
    ax1.axis('off')

    # Right subplot for reconstructed image
    ax2 = plt.subplot(gs[1])
    ax2.set_title("Reconstructed from Quaternion")
    ax2.axis('off')

    # Initialize with empty images
    img1 = ax1.imshow(np.zeros((image_size, image_size, 3), dtype=np.uint8))
    img2 = ax2.imshow(np.zeros((image_size, image_size, 3), dtype=np.uint8))

    # Add a text annotation for the sample index
    text = plt.figtext(0.5, 0.01, "", ha="center", fontsize=12)

    plt.tight_layout()
    sample_path = os.path.join(data_dir, sample_dir)

    # Load original image
    img_path = os.path.join(sample_path, "image.png")
    orig_img = np.array(Image.open(img_path))

    # Load quaternion
    quat_path = os.path.join(sample_path, "quaternion.txt")
    quaternion = np.loadtxt(quat_path)

    # Convert quaternion from (w, x, y, z) to scipy's (x, y, z, w) format
    scipy_quat = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rotation = Rotation.from_quat(scipy_quat)

    # Render the mesh with the rotation
    reconstructed_img = render_mesh_with_vectors(
        stl_mesh=stl_mesh,
        rotations=[rotation],
        image_size=image_size,
        show=False,
        show_vec=True
    )

    # Update the images
    img1.set_array(orig_img)
    img2.set_array(reconstructed_img)

    # Update the text
    text.set_text(f"Sample {sample_dir}")

    plt.show()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate diverse dataset from STL file')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for renders')
    parser.add_argument('--num_samples', type=int, default=5000, help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256, help='Image resolution')
    parser.add_argument('--rotation_method', type=str, default='fibonacci',
                      choices=['random', 'sphere', 'grid', 'fibonacci'],
                      help='Method for generating rotations')
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize the dataset after generation with side-by-side comparisons')
    parser.add_argument('--vis_samples', type=int, default=20,
                      help='Number of samples to visualize (default: 20)')
    parser.add_argument('--pause_time', type=float, default=3.0,
                      help='Time to pause on each sample during visualization (in seconds)')

    args = parser.parse_args()

    print(f"\nStarting diverse dataset generation:")
    print(f"STL file: {args.stl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Image resolution: {args.image_size}x{args.image_size}")
    print(f"Rotation method: {args.rotation_method}")

    # Load mesh
    print("Loading and normalizing mesh...")
    stl_mesh = load_and_normalize_stl(args.stl_path)

    # Check if we're only visualizing an existing dataset
    if args.visualize and os.path.exists(args.output_dir) and os.path.isdir(args.output_dir):
        # Check if there are sample directories in the output directory
        sample_dirs = [d for d in os.listdir(args.output_dir)
                      if os.path.isdir(os.path.join(args.output_dir, d)) and d.isdigit()]

        if len(sample_dirs) > 0:
            visualize_only = input(f"Dataset already exists at {args.output_dir}. Visualize without regenerating? (y/n): ")
            if visualize_only.lower() == 'y':
                print(f"Skipping dataset generation, visualizing sample from existing dataset...")
                visualize_sample(
                    stl_mesh=stl_mesh,
                    data_dir=args.output_dir,
                    sample_idx=0,
                    image_size=args.image_size,
                )
                return

    # Generate dataset
    print("Generating dataset...")
    start_time = time.time()
    generate_dataset(
        stl_mesh,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        image_size=args.image_size,
        rotation_method=args.rotation_method,
        visualize=args.visualize
    )

    # Copy the STL file into the generated data directory for bookkeeping.
    shutil.copy(args.stl_path, os.path.join(args.output_dir, "model.stl"))

    total_time = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {args.num_samples/total_time:.1f} images/second")
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
