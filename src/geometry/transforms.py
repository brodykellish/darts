import numpy as np
import torch
from scipy.spatial.transform import Rotation

class Transform3D:
    """A unified 3D transformation class supporting various rotation representations."""
    
    @staticmethod
    def from_euler(euler, translation=None):
        """Create transform from Euler angles (x,y,z) in radians."""
        rot = Rotation.from_euler('xyz', euler, degrees=False)
        return Transform3D(rot, translation)
    
    @staticmethod
    def from_quaternion(quat, translation=None):
        """Create transform from quaternion [x,y,z,w]."""
        rot = Rotation.from_quat(quat)
        return Transform3D(rot, translation)
    
    @staticmethod
    def from_matrix(matrix):
        """Create transform from 4x4 transformation matrix."""
        rot = Rotation.from_matrix(matrix[:3, :3])
        translation = matrix[:3, 3] if matrix.shape[0] >= 4 else None
        return Transform3D(rot, translation)
    
    def __init__(self, rotation=None, translation=None):
        self.rotation = rotation if rotation is not None else Rotation.identity()
        self.translation = np.zeros(3) if translation is None else np.array(translation)
    
    def as_matrix(self):
        """Return 4x4 transformation matrix."""
        result = np.eye(4)
        result[:3, :3] = self.rotation.as_matrix()
        result[:3, 3] = self.translation
        return result
    
    def as_euler(self):
        """Return Euler angles in radians."""
        return self.rotation.as_euler('xyz')
    
    def as_quaternion(self):
        """Return quaternion [x,y,z,w]."""
        return self.rotation.as_quat()
    
    def compose(self, other):
        """Compose this transform with another."""
        new_rot = self.rotation * other.rotation
        new_trans = self.translation + self.rotation.apply(other.translation)
        return Transform3D(new_rot, new_trans)
    
    def inverse(self):
        """Return the inverse transformation."""
        inv_rot = self.rotation.inv()
        inv_trans = -inv_rot.apply(self.translation)
        return Transform3D(inv_rot, inv_trans)
    
    def apply_to_point(self, point):
        """Apply transform to a 3D point."""
        return self.rotation.apply(point) + self.translation