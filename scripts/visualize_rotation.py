#!/usr/bin/env python3
"""
Script to visualize rotations of an STL model.
This can be used to:
1. Visualize a single rotation
2. Compare true and predicted rotations
3. Visualize a sequence of rotations from a dataset
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.render.stl_renderer import load_stl, normalize_mesh, render_mesh_with_true_pred

def visualize_single_rotation(args):
    """Visualize a single rotation of an STL model."""
    # Load and normalize the mesh
    stl_mesh = load_stl(args.stl_path)
    stl_mesh = normalize_mesh(stl_mesh)

    # Parse rotations
    true_rotation = np.array([float(x) for x in args.true_rotation.split(',')])
    pred_rotation = np.array([float(x) for x in args.pred_rotation.split(',')])

    # Convert from degrees to radians if needed
    if args.degrees:
        true_rotation = np.radians(true_rotation)
        pred_rotation = np.radians(pred_rotation)

    # Render the mesh
    output_path = args.output_path if args.output_path else None
    render_mesh_with_true_pred(stl_mesh, true_rotation=true_rotation, pred_rotation=pred_rotation,
                image_size=args.image_size, output_path=output_path, show=True,
                true_color=args.true_color, pred_color=args.pred_color, vec_scale=args.vec_scale)

    print(f"True Rotation (radians): {true_rotation}")
    print(f"Predicted Rotation (radians): {pred_rotation}")
    if args.degrees:
        print(f"True Rotation (degrees): {np.degrees(true_rotation)}")
        print(f"Predicted Rotation (degrees): {np.degrees(pred_rotation)}")

def main():
    parser = argparse.ArgumentParser(description='Visualize rotations of an STL model')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Single rotation visualization
    single_parser = subparsers.add_parser('single', help='Visualize a single rotation')
    single_parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    single_parser.add_argument('--true_rotation', type=str, required=True, help='True rotation as comma-separated Euler angles (x,y,z)')
    single_parser.add_argument('--pred_rotation', type=str, required=True, help='Predicted rotation as comma-separated Euler angles (x,y,z)')
    single_parser.add_argument('--degrees', action='store_true', help='Interpret rotation angles as degrees instead of radians')
    single_parser.add_argument('--true_color', type=str, default='red', help='Color of the true rotation vector')
    single_parser.add_argument('--pred_color', type=str, default='blue', help='Color of the predicted rotation vector')
    single_parser.add_argument('--vec_scale', type=float, default=0.7, help='Scale factor for the rotation vector (as fraction of max range)')
    single_parser.add_argument('--image_size', type=int, default=256, help='Image size')
    single_parser.add_argument('--output_path', type=str, help='Path to save the output image')

    args = parser.parse_args()

    if args.command == 'single':
        visualize_single_rotation(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
