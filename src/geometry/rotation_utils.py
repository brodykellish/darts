import numpy as np
from scipy.spatial.transform import Rotation

def generate_uniform_rotations(num_samples, method='sphere'):
    """
    Generate a set of rotations with a more uniform distribution.
    
    Args:
        num_samples: Number of rotations to generate
        method: Sampling method ('random', 'sphere', 'grid', 'fibonacci')
        
    Returns:
        List of Rotation objects
    """
    if method == 'random':
        # Standard random sampling (not uniform)
        return [Rotation.random() for _ in range(num_samples)]
    
    elif method == 'sphere':
        # Uniform sampling on a sphere
        rotations = []
        for _ in range(num_samples):
            # Sample a random point on the unit sphere
            v = np.random.randn(3)
            v = v / np.linalg.norm(v)
            
            # Random rotation around this axis
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Create rotation from axis-angle
            rot = Rotation.from_rotvec(angle * v)
            rotations.append(rot)
        
        return rotations
    
    elif method == 'grid':
        # Grid-based sampling (more uniform but not perfect)
        rotations = []
        
        # Determine grid size based on number of samples
        grid_size = int(np.ceil(np.cbrt(num_samples)))
        
        # Generate grid points
        for i in range(grid_size):
            for j in range(grid_size):
                for k in range(grid_size):
                    if len(rotations) >= num_samples:
                        break
                    
                    # Convert grid indices to Euler angles
                    alpha = 2 * np.pi * i / grid_size
                    beta = np.pi * j / (grid_size - 1) - np.pi/2
                    gamma = 2 * np.pi * k / grid_size
                    
                    # Create rotation from Euler angles
                    rot = Rotation.from_euler('xyz', [alpha, beta, gamma])
                    rotations.append(rot)
        
        # If we have too many rotations, randomly select the required number
        if len(rotations) > num_samples:
            indices = np.random.choice(len(rotations), num_samples, replace=False)
            rotations = [rotations[i] for i in indices]
        
        return rotations
    
    elif method == 'fibonacci':
        # Fibonacci sphere method (very uniform)
        rotations = []
        
        # Golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        for i in range(num_samples):
            # Fibonacci sphere formula
            y = 1 - (2 * i) / (num_samples - 1)
            radius = np.sqrt(1 - y * y)
            
            theta = 2 * np.pi * i / phi
            
            x = radius * np.cos(theta)
            z = radius * np.sin(theta)
            
            # Create a point on the sphere
            point = np.array([x, y, z])
            
            # Random rotation around this axis
            axis = point / np.linalg.norm(point)
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Create rotation from axis-angle
            rot = Rotation.from_rotvec(angle * axis)
            rotations.append(rot)
        
        return rotations
    
    else:
        raise ValueError(f"Unknown sampling method: {method}")