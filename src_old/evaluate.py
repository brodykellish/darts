import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from train import DartDataset, PosePredictor, get_device
import argparse
import json
from datetime import datetime

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model(model_path, device):
    model = PosePredictor()
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def load_test_data(data_dir, test_split=0.2, subset_size=5):
    dataset = DartDataset(data_dir)
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    test_subset = torch.utils.data.Subset(test_dataset, range(min(subset_size, len(test_dataset))))
    return test_subset

def quat_to_rotmat(q):
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = q
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    return R

def compute_metrics(R_gt, t_gt, R_pred, t_pred):
    """Compute rotation and translation errors."""
    print(f"[DEBUG] R_gt shape: {R_gt.shape}, R_pred shape: {R_pred.shape}")
    print(f"[DEBUG] R_gt: {R_gt}")
    print(f"[DEBUG] R_pred: {R_pred}")
    # Convert quaternions to rotation matrices if needed
    if R_gt.shape == (4,):
        R_gt = quat_to_rotmat(R_gt)
    if R_pred.shape == (4,):
        R_pred = quat_to_rotmat(R_pred)
    # Rotation error (in degrees)
    R_error = np.arccos((np.trace(R_gt.T @ R_pred) - 1) / 2) * 180 / np.pi
    # Translation error (Euclidean distance)
    t_error = np.linalg.norm(t_gt - t_pred)
    return {
        'rotation_error_deg': float(R_error),
        'translation_error': float(t_error)
    }

def save_test_case(output_dir, index, image, R_gt, t_gt, R_pred, t_pred, metrics):
    """Save a test case with all relevant information."""
    case_dir = os.path.join(output_dir, f'test_case_{index:03d}')
    os.makedirs(case_dir, exist_ok=True)
    
    # Save input image
    plt.imsave(os.path.join(case_dir, 'input.png'), image.permute(1, 2, 0).cpu().numpy())
    
    # Ensure R_gt and R_pred are 3x3 matrices
    if R_gt.shape == (4,):
        R_gt = quat_to_rotmat(R_gt)
    if R_pred.shape == (4,):
        R_pred = quat_to_rotmat(R_pred)
    
    # Save ground truth and predicted poses
    np.savetxt(os.path.join(case_dir, 'pose_gt.txt'), np.hstack((R_gt, t_gt.reshape(3, 1))))
    np.savetxt(os.path.join(case_dir, 'pose_pred.txt'), np.hstack((R_pred, t_pred.reshape(3, 1))))
    
    # Save metrics
    with open(os.path.join(case_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions and generate test cases')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--output-dir', type=str, default='evaluation', help='Directory to save evaluation results')
    parser.add_argument('--num-test-cases', type=int, default=5, help='Number of test cases to generate')
    args = parser.parse_args()

    # Setup
    device = get_device()
    model = load_model(args.model_path, device)
    test_subset = load_test_data(args.data_dir, subset_size=args.num_test_cases)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for this evaluation run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(eval_dir, exist_ok=True)
    
    # Collect metrics for all test cases
    all_metrics = []
    
    # Process each test case
    for i, (img, R_gt, t_gt) in enumerate(test_subset):
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            if isinstance(outputs, tuple) and len(outputs) == 3:
                _, R_pred, t_pred = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                R_pred, t_pred = outputs
            else:
                raise ValueError('Unexpected model output format')
        # Convert to numpy for metrics computation
        R_pred = R_pred.cpu().numpy()[0]
        t_pred = t_pred.cpu().numpy()[0]
        R_gt = R_gt.numpy()
        t_gt = t_gt.numpy()
        # Compute metrics
        metrics = compute_metrics(R_gt, t_gt, R_pred, t_pred)
        all_metrics.append(metrics)
        # Save test case
        save_test_case(eval_dir, i, img[0], R_gt, t_gt, R_pred, t_pred, metrics)
    
    # Compute and save overall metrics
    avg_metrics = {
        'avg_rotation_error_deg': np.mean([m['rotation_error_deg'] for m in all_metrics]),
        'avg_translation_error': np.mean([m['translation_error'] for m in all_metrics]),
        'std_rotation_error_deg': np.std([m['rotation_error_deg'] for m in all_metrics]),
        'std_translation_error': np.std([m['translation_error'] for m in all_metrics])
    }
    
    with open(os.path.join(eval_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {eval_dir}")
    print(f"Average rotation error: {avg_metrics['avg_rotation_error_deg']:.2f}°")
    print(f"Average translation error: {avg_metrics['avg_translation_error']:.4f}")

if __name__ == "__main__":
    main() 