import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from train import DartDataset, PosePredictor, get_device
import argparse

def quat2rotmat(q):
    # Convert quaternion to rotation matrix
    q = q / torch.norm(q, dim=-1, keepdim=True)
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    R = torch.stack([
        1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw,
        2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw,
        2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2
    ], dim=-1).reshape(-1, 3, 3)
    return R

def main():
    parser = argparse.ArgumentParser(description='Run model and save predictions')
    parser.add_argument('--model-path', type=str, required=True,
                      help='Path to the trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=5,
                      help='Number of samples to process')
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    # Load model
    model = PosePredictor().to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Create dataset and dataloader
    dataset = DartDataset("./dart_dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Get predictions
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, q_gt, t_gt in dataloader:
            images = images.to(device)
            q_gt = q_gt.to(device)
            t_gt = t_gt.to(device)
            
            # Get predictions
            x_recon, q_pred, t_pred = model(images)
            
            # Convert quaternions to rotation matrices
            R_gt = quat2rotmat(q_gt)
            R_pred = quat2rotmat(q_pred)
            
            # Store predictions
            for i in range(min(args.num_samples, len(images))):
                predictions.append({
                    'R_pred': R_pred[i].cpu().numpy().tolist(),
                    't_pred': t_pred[i].cpu().numpy().tolist(),
                    'R_gt': R_gt[i].cpu().numpy().tolist(),
                    't_gt': t_gt[i].cpu().numpy().tolist()
                })
            
            if len(predictions) >= args.num_samples:
                break
    
    # Save predictions
    os.makedirs('temp', exist_ok=True)
    with open('temp/predictions.json', 'w') as f:
        json.dump(predictions, f)
    
    print(f"Saved {len(predictions)} predictions to temp/predictions.json")

if __name__ == "__main__":
    main() 