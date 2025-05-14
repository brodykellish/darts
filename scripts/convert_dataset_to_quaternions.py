#!/usr/bin/env python3
"""
Script to convert an existing dataset with Euler angles to quaternions.
This script:
1. Loads a dataset with Euler angles
2. Converts the Euler angles to quaternions
3. Saves the quaternions alongside the original Euler angles
"""

import argparse
import os
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.transform import Rotation

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def convert_dataset(data_dir, output_dir=None, overwrite=False):
    """
    Convert a dataset with Euler angles to quaternions.
    
    Args:
        data_dir (str): Directory containing the dataset
        output_dir (str, optional): Directory to save the converted dataset
        overwrite (bool): Whether to overwrite existing quaternion files
    """
    # If no output directory is specified, use the input directory
    if output_dir is None:
        output_dir = data_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all example directories
    example_dirs = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"Converting {len(example_dirs)} examples...")
    
    for ex_dir in tqdm(example_dirs):
        # Get paths
        input_path = os.path.join(data_dir, ex_dir)
        output_path = os.path.join(output_dir, ex_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Check if quaternion file already exists
        quat_path = os.path.join(output_path, "quaternion.txt")
        if os.path.exists(quat_path) and not overwrite:
            continue
        
        # Load Euler angles
        rot_path = os.path.join(input_path, "rotation.txt")
        if not os.path.exists(rot_path):
            print(f"Warning: No rotation.txt found in {input_path}")
            continue
            
        euler_angles = np.loadtxt(rot_path)
        
        # Convert to quaternion
        rotation = Rotation.from_euler('xyz', euler_angles)
        quaternion = rotation.as_quat()  # Returns (x, y, z, w)
        
        # Reorder to (w, x, y, z) which is more common
        quaternion = np.array([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])
        
        # Save quaternion
        np.savetxt(quat_path, quaternion)
        
        # If output directory is different from input, copy the image
        if output_dir != data_dir:
            import shutil
            img_path = os.path.join(input_path, "image.png")
            if os.path.exists(img_path):
                shutil.copy(img_path, os.path.join(output_path, "image.png"))
            
            # Also copy the original rotation file
            shutil.copy(rot_path, os.path.join(output_path, "rotation.txt"))
    
    print(f"Conversion complete. Quaternions saved to {output_dir}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert dataset from Euler angles to quaternions')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save the converted dataset')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing quaternion files')
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_dataset(args.data_dir, args.output_dir, args.overwrite)


if __name__ == "__main__":
    main()
