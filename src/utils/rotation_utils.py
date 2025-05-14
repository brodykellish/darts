import numpy as np
import torch
from scipy.spatial.transform import Rotation

def euler_to_quaternion(euler_angles):
    """Convert Euler angles to quaternion."""
    r = Rotation.from_euler('xyz', euler_angles, degrees=False)
    return r.as_quat()  # Returns x, y, z, w

def quaternion_to_euler(quaternion):
    """Convert quaternion to Euler angles."""
    r = Rotation.from_quat(quaternion)
    return r.as_euler('xyz', degrees=False)

def quaternion_loss(pred, target):
    """Compute distance between two quaternions."""
    # Normalize quaternions
    pred_normalized = pred / torch.norm(pred, dim=1, keepdim=True)
    target_normalized = target / torch.norm(target, dim=1, keepdim=True)
    
    # Compute dot product
    dot_product = torch.sum(pred_normalized * target_normalized, dim=1)
    
    # Handle double cover: q and -q represent the same rotation
    dot_product = torch.abs(dot_product)
    
    # Compute angular distance: arccos(|q1·q2|)
    angle = 2 * torch.acos(torch.clamp(dot_product, -1.0, 1.0))
    
    return angle.mean()