import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RotationLoss(nn.Module):
    """
    Custom loss function for rotation prediction that handles the cyclical nature of rotations.
    This loss can work with both Euler angles and quaternions.
    """
    def __init__(self, mode='quaternion', reduction='mean'):
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

class DirectionalRotationLoss(nn.Module):
    """
    Custom loss function for rotation prediction that handles the cyclical nature of rotations.
    This loss can work with both Euler angles and quaternions.
    """
    def __init__(self, rotation_mode='quaternion', reduction='mean'):
        """
        Args:
            mode (str): 'euler' or 'quaternion' - determines the rotation representation
            reduction (str): 'mean', 'sum', or 'none' - reduction method
        """
        super().__init__()
        self.rotation_mode = rotation_mode
        self.reduction = reduction

    def forward(self, pred, target):
        if self.rotation_mode == 'euler':
            return self.euler_loss(pred, target)
        elif self.rotation_mode == 'quaternion':
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
        Compute loss for quaternions, focusing only on the 3D directional component.
        This loss ignores rotation around the axis and only considers the direction
        that a reference vector (e.g., [0, 0, 1]) would point after rotation.

        Args:
            pred (torch.Tensor): Predicted quaternions, shape (B, 4)
            target (torch.Tensor): Target quaternions, shape (B, 4)

        Returns:
            torch.Tensor: Loss value
        """
        # Normalize quaternions to unit length
        pred = F.normalize(pred, p=2, dim=1)
        target = F.normalize(target, p=2, dim=1)

        # Define a reference vector (using z-axis as default)
        # This is the vector whose direction we care about after rotation
        ref_vector = torch.tensor([0.0, 0.0, 1.0], device=pred.device)

        # Apply quaternion rotation to the reference vector for both predicted and target
        # For each quaternion in the batch
        batch_size = pred.shape[0]
        pred_vectors = []
        target_vectors = []

        for i in range(batch_size):
            # Extract quaternion components (assuming quaternion format is [w, x, y, z])
            p_w, p_x, p_y, p_z = pred[i]
            t_w, t_x, t_y, t_z = target[i]

            # Apply quaternion rotation to reference vector for prediction
            # Formula: v' = qvq* where v is [0,x,y,z] and q* is conjugate of q
            # Simplified formula for rotating a vector by a quaternion:
            p_vec = self._rotate_vector_by_quaternion(ref_vector, (p_w, p_x, p_y, p_z))
            t_vec = self._rotate_vector_by_quaternion(ref_vector, (t_w, t_x, t_y, t_z))

            pred_vectors.append(p_vec)
            target_vectors.append(t_vec)

        # Stack vectors into tensors
        pred_vectors = torch.stack(pred_vectors)
        target_vectors = torch.stack(target_vectors)

        # Normalize the rotated vectors (should already be unit length, but for numerical stability)
        pred_vectors = F.normalize(pred_vectors, p=2, dim=1)
        target_vectors = F.normalize(target_vectors, p=2, dim=1)

        # Compute the cosine similarity between the rotated vectors
        # This measures how closely the directions align
        cos_sim = torch.sum(pred_vectors * target_vectors, dim=1)

        # Convert to angle (in radians)
        # cos_sim is clamped to [-1, 1] to avoid numerical issues with acos
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
        angle = torch.acos(cos_sim)

        # Square the angle (similar to MSE)
        squared_angle = angle ** 2

        # Apply reduction
        if self.reduction == 'mean':
            return squared_angle.mean()
        elif self.reduction == 'sum':
            return squared_angle.sum()
        else:  # 'none'
            return squared_angle

    def _rotate_vector_by_quaternion(self, vector, quaternion):
        """
        Rotate a 3D vector by a quaternion.

        Args:
            vector (torch.Tensor): 3D vector to rotate [x, y, z]
            quaternion (tuple): Quaternion as (w, x, y, z)

        Returns:
            torch.Tensor: Rotated vector
        """
        w, x, y, z = quaternion

        # Compute the rotated vector using the quaternion rotation formula
        # v' = qvq* (where v is the vector and q* is the conjugate of q)

        # Simplified formula:
        # v' = v + 2w(q_v × v) + 2(q_v × (q_v × v))
        # where q_v is the vector part of the quaternion [x, y, z]

        # Vector part of quaternion
        q_v = torch.tensor([x, y, z], device=vector.device)

        # First term: v
        term1 = vector

        # Second term: 2w(q_v × v)
        cross1 = torch.cross(q_v, vector)
        term2 = 2 * w * cross1

        # Third term: 2(q_v × (q_v × v))
        cross2 = torch.cross(q_v, cross1)
        term3 = 2 * cross2

        # Combine terms
        rotated_vector = term1 + term2 + term3

        return rotated_vector