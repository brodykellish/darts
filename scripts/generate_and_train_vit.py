#!/usr/bin/env python3
"""
Script to generate a sample dataset and train the ViTPosePredictor model.
This script:
1. Generates a sample dataset using the simple data generator
2. Trains a ViTPosePredictor model on the generated dataset
3. Saves the trained model and training logs
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def generate_dataset(stl_path, output_dir, num_samples=1000, image_size=256):
    """
    Generate a dataset using the simple data generator script.
    
    Args:
        stl_path (str): Path to the STL file
        output_dir (str): Directory to save the dataset
        num_samples (int): Number of samples to generate
        image_size (int): Size of the rendered images
    """
    print(f"\n=== Generating dataset with {num_samples} samples ===")
    
    # Create the command
    cmd = [
        "python", 
        "scripts/generate_simple_dataset.py",
        "--stl_path", stl_path,
        "--output_dir", output_dir,
        "--num_samples", str(num_samples),
        "--image_size", str(image_size)
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Dataset generation complete. Saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating dataset: {e}")
        sys.exit(1)


def train_model(data_dir, output_dir, epochs=50, batch_size=32, image_size=224):
    """
    Train the ViTPosePredictor model on the generated dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
        output_dir (str): Directory to save the model checkpoints
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        image_size (int): Size to resize images to (ViT typically uses 224)
    """
    print(f"\n=== Training ViTPosePredictor for {epochs} epochs ===")
    
    # Create the command
    cmd = [
        "python", 
        "scripts/train_vit_pose_predictor.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--image_size", str(image_size),
        "--log_interval", "5"
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Training complete. Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error training model: {e}")
        sys.exit(1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate dataset and train ViTPosePredictor')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--data_dir', type=str, default='./vit_dataset', help='Directory to save the dataset')
    parser.add_argument('--output_dir', type=str, default='./vit_model', help='Directory to save model checkpoints')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--skip_generation', action='store_true', help='Skip dataset generation')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset if not skipped
    if not args.skip_generation:
        generate_dataset(
            stl_path=args.stl_path,
            output_dir=args.data_dir,
            num_samples=args.num_samples,
            image_size=256  # Generate at 256x256, will be resized during training
        )
    else:
        print("Skipping dataset generation.")
    
    # Train model
    train_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=224  # ViT typically uses 224x224
    )
    
    print("\n=== Process complete ===")
    print(f"Dataset: {args.data_dir}")
    print(f"Trained model: {args.output_dir}")


if __name__ == "__main__":
    main()
