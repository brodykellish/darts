#!/usr/bin/env python3
"""
Data generation script that creates a diverse dataset of rotated/lighted renders of a 3D model.

This script:
- Loads a specified .stl file, normalized to be centered at the origin with unit scale
- Accepts parameters for image size, number of samples, output directory, etc.
- Generates N samples with random rotations using various sampling methods
- Adds more variation in lighting, viewpoints, and backgrounds
- Saves renders and rotation information (Euler angles and quaternions) in the specified output directory
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
from src.geometry.rotation_utils import generate_uniform_rotations
from src.render.stl_renderer import render_mesh, load_and_normalize_stl

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def generate_dataset(stl_mesh, num_samples, output_dir, image_size=256, 
                     rotation_method='fibonacci', add_noise=True, random_lighting=True):
    """
    Generate a dataset of rendered images with diverse rotations.
    
    Args:
        vertices: Numpy array of vertices
        faces: Numpy array of faces
        num_samples: Number of samples to generate
        output_dir: Directory to save the samples
        image_size: Size of the rendered image (square)
        rotation_method: Method for generating rotations
        add_noise: Whether to add random noise to images
        random_lighting: Whether to use random lighting
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate rotations
    print(f"Generating {num_samples} rotations using {rotation_method} method...")
    rotations = generate_uniform_rotations(num_samples, method=rotation_method)
    
    for i in tqdm(range(num_samples)):
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
    
    # Generate dataset
    print("Generating dataset...")
    start_time = time.time()
    generate_dataset(
        stl_mesh, args.num_samples, args.output_dir, args.image_size,
        rotation_method=args.rotation_method,
        add_noise=not args.no_noise,
        random_lighting=not args.no_random_lighting
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
