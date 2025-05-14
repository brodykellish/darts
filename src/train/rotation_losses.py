import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotationLoss(nn.Module):
    """
    Custom loss function for rotation prediction that handles the cyclical nature of rotations.
    This loss can work with both Euler angles and quaternions.
    """
    def __init__(self, mode='euler', reduction='mean'):
        """
        Args:
            mode (str): 'euler' or 'quaternion' - determines the rotation representation
            reduction (str): 'mean', 'sum', or 'none' - reduction method
        """
        super().__init__()
        self.mode = mode
        self.reduction = reduction
        
    def forward(self, pred, target):
        if self.mode == 'euler':
            return self.euler_loss(pred, target)
        elif self.mode == 'quaternion':
            return self.quaternion_loss(pred, target)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
    
    def euler_loss(self, pred, target):
        """
        Compute loss for Euler angles, accounting for their cyclical nature.
        
        Args:
            pred (torch.Tensor): Predicted Euler angles in radians, shape (B, 3)
            target (torch.Tensor): Target Euler angles in radians, shape (B, 3)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Handle the cyclical nature of angles
        # For each angle, compute the minimum distance considering the 2π periodicity
        angle_diff = torch.abs(pred - target)
        angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)
        
        # Square the differences (similar to MSE)
        squared_diff = angle_diff ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            return squared_diff.mean()
        elif self.reduction == 'sum':
            return squared_diff.sum()
        else:  # 'none'
            return squared_diff
    
    def quaternion_loss(self, pred, target):
        """
        Compute loss for quaternions, using the geodesic distance on the unit sphere.
        
        Args:
            pred (torch.Tensor): Predicted quaternions, shape (B, 4)
            target (torch.Tensor): Target quaternions, shape (B, 4)
            
        Returns:
            torch.Tensor: Loss value
        """
        # Normalize quaternions to unit length
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)
        
        # Compute the dot product between quaternions
        # The absolute value handles the fact that q and -q represent the same rotation
        dot_product = torch.abs(torch.sum(pred * target, dim=1))
        
        # Clamp to avoid numerical issues with acos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Compute the geodesic distance (angle) between quaternions
        angle = 2 * torch.acos(dot_product)
        
        # Square the angle (similar to MSE)
        squared_angle = angle ** 2
        
        # Apply reduction
        if self.reduction == 'mean':
            return squared_angle.mean()
        elif self.reduction == 'sum':
            return squared_angle.sum()
        else:  # 'none'
            return squared_angle


class CombinedRotationTranslationLoss(nn.Module):
    """
    Combined loss for both rotation and translation prediction.
    """
    def __init__(self, rotation_mode='euler', rotation_weight=1.0, translation_weight=0.1):
        """
        Args:
            rotation_mode (str): 'euler' or 'quaternion' - determines the rotation representation
            rotation_weight (float): Weight for the rotation loss
            translation_weight (float): Weight for the translation loss
        """
        super().__init__()
        self.rotation_loss = RotationLoss(mode=rotation_mode)
        self.translation_loss = nn.MSELoss()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        
    def forward(self, pred_rot, pred_trans, target_rot, target_trans):
        """
        Compute the combined loss.
        
        Args:
            pred_rot (torch.Tensor): Predicted rotations
            pred_trans (torch.Tensor): Predicted translations
            target_rot (torch.Tensor): Target rotations
            target_trans (torch.Tensor): Target translations
            
        Returns:
            torch.Tensor: Combined loss value
            dict: Dictionary with individual loss components
        """
        rot_loss = self.rotation_loss(pred_rot, target_rot)
        trans_loss = self.translation_loss(pred_trans, target_trans)
        
        # Combine losses
        combined_loss = self.rotation_weight * rot_loss + self.translation_weight * trans_loss
        
        # Return both the combined loss and individual components
        return combined_loss, {
            'rotation_loss': rot_loss.item(),
            'translation_loss': trans_loss.item(),
            'combined_loss': combined_loss.item()
        }
