#!/usr/bin/env python3
"""
Simple data generation script for rendering 3D models from STL files.
This script:
- Loads a specified .stl file, normalized to be centered at the origin with unit scale
- Accepts parameters for image size, number of samples, output directory, etc.
- Generates N samples with random rotations
- Saves renders and rotation information in the specified output directory

This is a simplified version that doesn't require PyTorch3D, using only
basic libraries like NumPy, Matplotlib, and is compatible with NumPy 2.0.
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
import numpy.lib.recfunctions as rfn
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def load_stl(file_path):
    """
    Load an STL file using NumPy's loadtxt.

    Args:
        file_path: Path to the STL file

    Returns:
        vertices and faces arrays
    """
    from stl import mesh

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(file_path)

    # Extract vertices and faces
    # STL files store triangles directly, so we need to extract unique vertices
    triangles = stl_mesh.vectors

    # Reshape to get all vertices
    vertices_all = triangles.reshape(-1, 3)

    # Get unique vertices
    # This is a simplified approach - in a real application, you might want
    # to use a more sophisticated method to handle numerical precision issues
    vertices, inverse = np.unique(vertices_all.round(decimals=5), axis=0, return_inverse=True)

    # Create faces from inverse indices
    faces = inverse.reshape(-1, 3)

    return vertices, faces


def normalize_mesh(vertices):
    """
    Normalize vertices to be centered at the origin with unit scale.

    Args:
        vertices: Numpy array of vertices

    Returns:
        Normalized vertices
    """
    # Center at origin
    center = np.mean(vertices, axis=0)
    vertices = vertices - center

    # Scale to unit size
    scale = np.max(np.abs(vertices))
    vertices = vertices / scale

    return vertices


def load_and_normalize_stl(file_path):
    """
    Load an STL file and normalize it to be centered at the origin with unit scale.

    Args:
        file_path: Path to the STL file

    Returns:
        vertices and faces arrays
    """
    vertices, faces = load_stl(file_path)
    vertices = normalize_mesh(vertices)
    return vertices, faces


def render_mesh(vertices, faces, rotation, image_size=256):
    """
    Render a mesh with the given rotation using matplotlib.

    Args:
        vertices: Numpy array of vertices
        faces: Numpy array of faces
        rotation: Rotation object from scipy.spatial.transform
        image_size: Size of the rendered image (square)

    Returns:
        RGB image as a numpy array
    """
    # Apply rotation to vertices
    rotated_vertices = rotation.apply(vertices)

    # Create a figure with the right size and no axes
    dpi = 100
    fig = plt.figure(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

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

    # Set equal aspect ratio and remove axes
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    # Set view angle
    ax.view_init(elev=30, azim=45)

    # Render the image
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Convert to numpy array
    # Get the dimensions of the figure canvas
    width, height = fig.canvas.get_width_height()

    # Convert to numpy array using buffer_rgba (modern approach)
    try:
        # For newer matplotlib versions
        buffer = fig.canvas.buffer_rgba()
        data = np.asarray(buffer)
        # Convert RGBA to RGB
        data = data[:, :, :3]
    except AttributeError:
        # Fallback for older matplotlib versions
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(height, width, 3)

    # Close the figure to free memory
    plt.close(fig)

    return data


def generate_dataset(vertices, faces, num_samples, output_dir, image_size=256):
    """
    Generate a dataset of rendered images with random rotations.

    Args:
        vertices: Numpy array of vertices
        faces: Numpy array of faces
        num_samples: Number of samples to generate
        output_dir: Directory to save the samples
        image_size: Size of the rendered image (square)
    """
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(num_samples)):
        # Create subdirectory for this sample
        sample_dir = os.path.join(output_dir, f"{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        # Generate random rotation
        rotation = Rotation.random()

        # Get Euler angles in radians
        euler_angles = rotation.as_euler('xyz')

        # Render image
        image = render_mesh(vertices, faces, rotation, image_size)

        # Save image
        # Ensure image is in the correct format for saving (0-255 uint8)
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

        # Save using PIL which is more reliable than plt.imsave
        Image.fromarray(image).save(os.path.join(sample_dir, "image.png"))

        # Save rotation data
        np.savetxt(os.path.join(sample_dir, "rotation.txt"), euler_angles)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate dataset from STL file (simple version)')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for renders')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--image_size', type=int, default=256, help='Image resolution')

    args = parser.parse_args()

    print(f"\nStarting dataset generation:")
    print(f"STL file: {args.stl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Image resolution: {args.image_size}x{args.image_size}\n")

    # Load mesh
    print("Loading and normalizing mesh...")
    vertices, faces = load_and_normalize_stl(args.stl_path)

    # Generate dataset
    print("Generating dataset...")
    start_time = time.time()
    generate_dataset(vertices, faces, args.num_samples, args.output_dir, args.image_size)

    total_time = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {args.num_samples/total_time:.1f} images/second")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
