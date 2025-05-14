#!/usr/bin/env python3
"""
Enhanced data generation script for creating a more diverse dataset of 3D model rotations.
This script:
- Generates a more uniform distribution of rotations using various sampling methods
- Saves both Euler angles and quaternion representations
- Adds more variation in lighting, viewpoints, and backgrounds
- Supports data augmentation during generation
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


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


def generate_uniform_rotations(num_samples, method='sphere'):
    """
    Generate a set of rotations with a more uniform distribution.
    
    Args:
        num_samples: Number of rotations to generate
        method: Sampling method ('random', 'sphere', 'grid', 'fibonacci')
        
    Returns:
        List of Rotation objects
    """
    if method == 'random':
        # Standard random sampling (not uniform)
        return [Rotation.random() for _ in range(num_samples)]
    
    elif method == 'sphere':
        # Uniform sampling on a sphere
        rotations = []
        for _ in range(num_samples):
            # Sample a random point on the unit sphere
            v = np.random.randn(3)
            v = v / np.linalg.norm(v)
            
            # Random rotation around this axis
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Create rotation from axis-angle
            rot = Rotation.from_rotvec(angle * v)
            rotations.append(rot)
        
        return rotations
    
    elif method == 'grid':
        # Grid-based sampling (more uniform but not perfect)
        rotations = []
        
        # Determine grid size based on number of samples
        grid_size = int(np.ceil(np.cbrt(num_samples)))
        
        # Generate grid points
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(rotations) >= num_samples:
                        break
                    
                    # Convert grid indices to Euler angles
                    alpha = 2 * np.pi * i / grid_size
                    beta = np.pi * j / (grid_size - 1) - np.pi/2
                    gamma = 2 * np.pi * k / grid_size
                    
                    # Create rotation from Euler angles
                    rot = Rotation.from_euler('xyz', [alpha, beta, gamma])
                    rotations.append(rot)
        
        # If we have too many rotations, randomly select the required number
        if len(rotations) > num_samples:
            indices = np.random.choice(len(rotations), num_samples, replace=False)
            rotations = [rotations[i] for i in indices]
        
        return rotations
    
    elif method == 'fibonacci':
        # Fibonacci sphere method (very uniform)
        rotations = []
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(num_samples):
            # Fibonacci sphere formula
            y = 1 - (2 * i) / (num_samples - 1)
            radius = np.sqrt(1 - y * y)
            
            theta = 2 * np.pi * i / phi
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            # Create a point on the sphere
            point = np.array([x, y, z])
            
            # Random rotation around this axis
            axis = point / np.linalg.norm(point)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Create rotation from axis-angle
            rot = Rotation.from_rotvec(angle * axis)
            rotations.append(rot)
        
        return rotations
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def render_mesh(vertices, faces, rotation, image_size=256, view_elevation=30, view_azimuth=45,
                add_noise=False, random_lighting=False):
    """
    Render a mesh with the given rotation using matplotlib.
    
    Args:
        vertices: Numpy array of vertices
        faces: Numpy array of faces
        rotation: Rotation object from scipy.spatial.transform
        image_size: Size of the rendered image (square)
        view_elevation: Camera elevation angle
        view_azimuth: Camera azimuth angle
        add_noise: Whether to add random noise to the image
        random_lighting: Whether to use random lighting
        
    Returns:
        RGB image as a numpy array
    """
    # Apply rotation to vertices
    rotated_vertices = rotation.apply(vertices)
    
    # Create a figure with the right size and no axes
    dpi = 100
    fig = plt.figure(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    
    # Set random lighting if requested
    if random_lighting:
        light_color = np.random.uniform(0.7, 1.0, 3)  # Random RGB
        mesh_color = np.random.uniform(0.3, 0.8, 3)  # Random RGB
    else:
        light_color = np.array([1.0, 1.0, 1.0])  # White light
        mesh_color = np.array([0.7, 0.7, 0.7])  # Light gray
    
    # Plot the mesh
    ax.plot_trisurf(
        rotated_vertices[:, 0],
        rotated_vertices[:, 1],
        rotated_vertices[:, 2],
        triangles=faces,
        color=mesh_color,
        edgecolor=None,
        linewidth=0.2,
        alpha=0.8
    )
    
    # Set equal aspect ratio and remove axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    
    # Set view angle (with some randomness if requested)
    if random_lighting:
        elev = view_elevation + np.random.uniform(-10, 10)
        azim = view_azimuth + np.random.uniform(-10, 10)
    else:
        elev = view_elevation
        azim = view_azimuth
    
    ax.view_init(elev=elev, azim=azim)
    
    # Render the image
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    
    # Convert to numpy array
    width, height = fig.canvas.get_width_height()
    
    try:
        buffer = fig.canvas.buffer_rgba()
        data = np.asarray(buffer)
        data = data[:, :, :3]  # Convert RGBA to RGB
    except AttributeError:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(height, width, 3)
    
    # Add noise if requested
    if add_noise:
        noise = np.random.normal(0, 5, data.shape).astype(np.int16)
        data = np.clip(data.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Close the figure to free memory
    plt.close(fig)
    
    return data


def generate_dataset(vertices, faces, num_samples, output_dir, image_size=256, 
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
        
        # Get Euler angles in radians
        euler_angles = rotation.as_euler('xyz')
        
        # Get quaternion (x, y, z, w)
        quaternion = rotation.as_quat()
        
        # Reorder to (w, x, y, z) which is more common
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        # Render image
        image = render_mesh(
            vertices, faces, rotation, image_size,
            add_noise=add_noise, random_lighting=random_lighting
        )
        
        # Save image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        Image.fromarray(image).save(os.path.join(sample_dir, "image.png"))
        
        # Save rotation data in both formats
        np.savetxt(os.path.join(sample_dir, "rotation.txt"), euler_angles)
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
    parser.add_argument('--no_noise', action='store_true', help='Disable random noise in images')
    parser.add_argument('--no_random_lighting', action='store_true', help='Disable random lighting')
    
    args = parser.parse_args()
    
    print(f"\nStarting diverse dataset generation:")
    print(f"STL file: {args.stl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Image resolution: {args.image_size}x{args.image_size}")
    print(f"Rotation method: {args.rotation_method}")
    print(f"Add noise: {not args.no_noise}")
    print(f"Random lighting: {not args.no_random_lighting}\n")
    
    # Load mesh
    print("Loading and normalizing mesh...")
    vertices, faces = load_and_normalize_stl(args.stl_path)
    
    # Generate dataset
    print("Generating dataset...")
    start_time = time.time()
    generate_dataset(
        vertices, faces, args.num_samples, args.output_dir, args.image_size,
        rotation_method=args.rotation_method,
        add_noise=not args.no_noise,
        random_lighting=not args.no_random_lighting
    )
    
    total_time = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {args.num_samples/total_time:.1f} images/second")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
