import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from term_image.image import ImageIterator, ImageSource
from io import BytesIO

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.model.autoencoder import Autoencoder

def euler_to_rotation_matrix(euler_angles):
    """Convert Euler angles to rotation matrix."""
    # Extract angles
    x, y, z = euler_angles
    
    # Calculate rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations
    R = Rz @ Ry @ Rx
    return R

def visualize_prediction(input_img, output_img, true_rot, pred_rot, reconstruction_loss, rotation_loss, use_terminal=False):
    """Visualize input, output, and rotation vectors."""
    # Convert tensors to numpy arrays
    input_img = input_img.cpu().numpy().transpose(1, 2, 0)
    output_img = output_img.cpu().numpy().transpose(1, 2, 0)
    
    # Denormalize images
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input_img = std * input_img + mean
    output_img = std * output_img + mean
    
    # Clip to valid range
    input_img = np.clip(input_img, 0, 1)
    output_img = np.clip(output_img, 0, 1)
    
    if use_terminal:
        # Convert numpy arrays to PIL Images
        input_pil = Image.fromarray((input_img * 255).astype(np.uint8))
        output_pil = Image.fromarray((output_img * 255).astype(np.uint8))
        
        # Resize for terminal display (make them smaller)
        terminal_size = (40, 40)  # Adjust based on your terminal size
        input_pil = input_pil.resize(terminal_size)
        output_pil = output_pil.resize(terminal_size)
        
        # Convert to bytes for term-image
        input_buffer = BytesIO()
        output_buffer = BytesIO()
        input_pil.save(input_buffer, format='PNG')
        output_pil.save(output_buffer, format='PNG')
        
        # Reset buffer positions
        input_buffer.seek(0)
        output_buffer.seek(0)
        
        # Display in terminal
        print("\nInput Image:")
        print("=" * 50)
        ImageIterator(ImageSource.from_file(input_buffer)).display()
        
        print("\nReconstructed Image:")
        print("=" * 50)
        ImageIterator(ImageSource.from_file(output_buffer)).display()
        
        print("\nLoss Information:")
        print("=" * 50)
        print(f"Reconstruction Loss: {reconstruction_loss:.6f}")
        print(f"Rotation Loss: {rotation_loss:.6f}")
        
    else:
        # Create figure for matplotlib visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot images
        ax1.imshow(input_img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        ax2.imshow(output_img)
        ax2.set_title('Reconstructed Image')
        ax2.axis('off')
        
        # Add rotation vectors
        def draw_rotation_vector(ax, rot, color, label):
            # Convert Euler angles to rotation matrix
            rot_matrix = euler_to_rotation_matrix(rot)
            
            # Get transformed Z-axis
            z_axis = np.array([0, 0, 1])
            transformed_z = rot_matrix @ z_axis
            
            # Draw vector
            ax.arrow(128, 128,  # Start at center
                     transformed_z[0] * 50, transformed_z[1] * 50,  # Direction
                     head_width=5, head_length=10, fc=color, ec=color,
                     label=label)
        
        # Draw true rotation vector on input image
        draw_rotation_vector(ax1, true_rot, 'red', 'True Rotation')
        
        # Draw predicted rotation vector on output image
        draw_rotation_vector(ax2, pred_rot, 'green', 'Predicted Rotation')
        
        # Add legends
        ax1.legend()
        ax2.legend()
        
        # Add title with loss info
        plt.suptitle(f'Reconstruction Loss: {reconstruction_loss:.6f}\nRotation Loss: {rotation_loss:.6f}')
        
        # Show plot
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run autoencoder inference on a single example')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--datum_dir', type=str, required=True, help='Directory containing the example (e.g. data/renders/0000)')
    parser.add_argument('--image_size', type=int, default=256, help='Size to resize images to')
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimension of latent space')
    parser.add_argument('--term', action='store_true', help='Display images in terminal instead of matplotlib window')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = Autoencoder(args.latent_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and preprocess image
    datum_dir = Path(args.datum_dir)
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    img_path = datum_dir / 'image.png'
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Load true rotation
    rot_path = datum_dir / 'rotation.txt'
    true_rotation = np.loadtxt(rot_path)
    true_rotation_tensor = torch.tensor(true_rotation, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        # Forward pass
        output, latent = model(image_tensor)
        
        # Get predicted rotation
        pred_rotation = model.rotation_head(latent)
        
        # Compute losses
        reconstruction_loss = nn.MSELoss()(output, image_tensor).item()
        rotation_loss = nn.MSELoss()(pred_rotation, true_rotation_tensor).item()
        
        # Convert tensors to numpy for visualization
        pred_rotation = pred_rotation[0].cpu().numpy()
    
    # Print results
    print("\nInference Results:")
    print("=" * 50)
    print(f"Reconstruction Loss: {reconstruction_loss:.6f}")
    print(f"Rotation Loss: {rotation_loss:.6f}")
    print("\nTrue Rotation (Euler angles):")
    print(f"X: {true_rotation[0]:.3f}")
    print(f"Y: {true_rotation[1]:.3f}")
    print(f"Z: {true_rotation[2]:.3f}")
    print("\nPredicted Rotation (Euler angles):")
    print(f"X: {pred_rotation[0]:.3f}")
    print(f"Y: {pred_rotation[1]:.3f}")
    print(f"Z: {pred_rotation[2]:.3f}")
    
    # Visualize results
    visualize_prediction(
        image_tensor[0],  # Remove batch dimension
        output[0],
        true_rotation,
        pred_rotation,
        reconstruction_loss,
        rotation_loss,
        use_terminal=args.term
    )

if __name__ == '__main__':
    main() 