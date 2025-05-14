"""
Utility for rendering rotated STL models with an option to display a unit vector
indicating the direction of the rotation. This is useful for visually evaluating
predicted rotations at inference time.

This renderer uses trisurf with fixed lighting for consistent visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
from scipy.spatial.transform import Rotation
from PIL import Image
import os
from stl import mesh

def load_stl_mesh(file_path):
    """
    Load an STL file using numpy-stl.

    Args:
        file_path: Path to the STL file

    Returns:
        A numpy-stl mesh object
    """
    return mesh.Mesh.from_file(file_path)

def normalize_mesh(stl_mesh):
    """
    Normalize a mesh to be centered at the origin with unit scale.

    Args:
        stl_mesh: A numpy-stl mesh object

    Returns:
        A normalized copy of the mesh
    """
    # Create a copy of the mesh
    normalized_mesh = mesh.Mesh(stl_mesh.data.copy())

    # Center at origin
    center = np.mean(normalized_mesh.vectors.reshape(-1, 3), axis=0)
    normalized_mesh.vectors -= center

    # Scale to unit size
    scale = np.max(np.abs(normalized_mesh.vectors))
    normalized_mesh.vectors /= scale

    return normalized_mesh

def load_and_normalize_stl(file_path):
    return normalize_mesh(load_stl_mesh(file_path))

def apply_rotation(stl_mesh, rotation):
    """
    Apply a rotation to a mesh.

    Args:
        stl_mesh: A numpy-stl mesh object
        rotation: A scipy.spatial.transform.Rotation object or Euler angles in radians

    Returns:
        A rotated copy of the mesh
    """
    # Create a copy of the mesh
    rotated_mesh = mesh.Mesh(stl_mesh.data.copy())

    # Convert Euler angles to Rotation object if needed
    if isinstance(rotation, np.ndarray) and rotation.shape == (3,):
        rotation = Rotation.from_euler('xyz', rotation)

    # Apply rotation to all vertices
    # The vectors attribute has shape (N, 3, 3) where N is the number of triangles
    # We need to reshape it to apply the rotation to each vertex
    original_shape = rotated_mesh.vectors.shape
    reshaped_vectors = rotated_mesh.vectors.reshape(-1, 3)
    rotated_vectors = rotation.apply(reshaped_vectors)
    rotated_mesh.vectors = rotated_vectors.reshape(original_shape)

    return rotated_mesh

def get_rotation_vector(rotation):
    """
    Get a unit vector representing the direction of the rotation.

    Args:
        rotation: A scipy.spatial.transform.Rotation object or Euler angles in radians

    Returns:
        A unit vector (3D) representing the direction of the rotation
    """
    # Convert Euler angles to Rotation object if needed
    if isinstance(rotation, np.ndarray) and rotation.shape == (3,):
        rotation = Rotation.from_euler('xyz', rotation)

    # Use the z-axis as the reference vector
    reference_vector = np.array([0, 0, 1])

    # Apply rotation to the reference vector
    rotated_vector = rotation.apply(reference_vector)

    # Reverse the vector
    rotated_vector = -rotated_vector

    return rotated_vector

def add_rotation_vector(ax, rotation, max_range, color='red', scale=0.7, label=None, add_sphere=True):
    """
    Add a rotation vector to a 3D axis.

    Args:
        ax: Matplotlib 3D axis
        rotation: A scipy.spatial.transform.Rotation object or Euler angles in radians
        max_range: Maximum range of the plot for scaling
        color: Color of the vector
        scale: Scale factor for the vector as a fraction of max_range
        label: Optional label for the vector (for legend)
        add_sphere: Whether to add a sphere at the origin

    Returns:
        The quiver object representing the vector
    """
    # Get the rotation vector
    rot_vec = get_rotation_vector(rotation)

    # Scale the vector to make it more prominent
    scale_factor = max_range * scale / np.linalg.norm(rot_vec)
    scaled_vec = rot_vec * scale_factor

    # Add a small sphere at the origin if requested
    if add_sphere:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        sphere_radius = max_range * 0.01  # 1% of max range
        x = sphere_radius * np.cos(u) * np.sin(v)
        y = sphere_radius * np.sin(u) * np.sin(v)
        z = sphere_radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.8)

    # Plot the vector as an arrow with increased size and prominence
    quiver = ax.quiver(0, 0, 0, scaled_vec[0], scaled_vec[1], scaled_vec[2],
                      color=color, arrow_length_ratio=0.05, linewidth=1, label=label)

    return quiver

def render_mesh_with_vectors(stl_mesh, rotations=None, image_size=256,
                       view_elev=30, view_azim=45, output_path=None, show=True, show_vec=False):
    """
    Render a mesh with multiple rotation vectors using trisurf with fixed lighting.

    Args:
        stl_mesh: A numpy-stl mesh object
        rotations: List of rotations (Rotation objects or Euler angles in radians)
        image_size: Size of the rendered image (square)
        view_elev: Elevation angle for the view
        view_azim: Azimuth angle for the view
        output_path: Path to save the rendered image (if None, the image is not saved)
        show: Whether to display the rendered image
        show_vec: Whether to display rotation vectors

    Returns:
        The rendered image as a numpy array
    """
    # Apply the first rotation to the mesh if provided
    if rotations and len(rotations) > 0:
        stl_mesh = apply_rotation(stl_mesh, rotations[0])

    # Create a figure with the right size and no axes
    dpi = 100
    fig = plt.figure(figsize=(image_size/dpi, image_size/dpi), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    # Extract the triangles from the mesh
    # The vectors attribute has shape (N, 3, 3) where N is the number of triangles
    triangles = stl_mesh.vectors

    # Extract vertices and faces for trisurf
    # We need to create a list of unique vertices and a list of faces that reference these vertices
    vertices = triangles.reshape(-1, 3)
    vertices, inverse = np.unique(vertices.round(decimals=5), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)

    # Set fixed lighting parameters
    ax.set_facecolor('white')
    plt.rcParams['axes.facecolor'] = 'white'

    # Configure lighting
    ax.set_box_aspect([1, 1, 1])

    # Set fixed lighting parameters
    mesh_color = np.array([0.7, 0.7, 0.7])  # Light gray
    
    # Plot the mesh
    ax.plot_trisurf(
        vertices[:, 0], 
        vertices[:, 1], 
        vertices[:, 2],
        triangles=faces,
        color=mesh_color,
        edgecolor=None,
        linewidth=0.2,
        shade=True,
        alpha=0.8
    )

    # Calculate the maximum range from the mesh vertices
    max_range = np.max(np.abs(vertices))

    # Add rotation vectors if provided
    if rotations and show_vec:
        # Generate colors
        # Use a colormap to generate distinct colors
        cmap = plt.cm.get_cmap('tab10', len(rotations))
        vec_colors = [cmap(i) for i in range(len(rotations))]

        # Generate labels
        vec_labels = ["Applied rotation" if i == 0 else f"Rotation {i+1}" for i in range(len(rotations))]

        # Add each rotation vector
        quivers = []
        for i, rotation in enumerate(rotations):
            quiver = add_rotation_vector(
                ax, rotation, max_range,
                color=vec_colors[i],
                scale=1.5,
                label=vec_labels[i],
                add_sphere=(i == 0)  # Only add sphere for the first vector
            )
            quivers.append(quiver)

        # Add a legend if there are multiple vectors
        if len(rotations) > 1:
            ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    # Set axis limits based on the mesh size
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Render the image
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Convert to numpy array
    width, height = fig.canvas.get_width_height()
    try:
        # For newer matplotlib versions
        buffer = fig.canvas.buffer_rgba()
        image = np.asarray(buffer)
        # Convert RGBA to RGB
        image = image[:, :, :3]
    except AttributeError:
        # Fallback for older matplotlib versions
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(height, width, 3)

    # Save the image if output_path is provided
    if output_path is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        # Save the image
        Image.fromarray(image).save(output_path)

    # Show the image if requested
    if show:
        plt.show()
    else:
        plt.close(fig)

    return image

def render_mesh(stl_mesh, rotation=None, image_size=256, show=False,
                view_elev=30, view_azim=45, output_path=None):
    """
    Render a mesh with optional rotation.

    Args:
        stl_mesh: A numpy-stl mesh object
        rotation: A scipy.spatial.transform.Rotation object or Euler angles in radians
        image_size: Size of the rendered image (square)
        show: Whether to display the rendered image
        view_elev: Elevation angle for the view
        view_azim: Azimuth angle for the view
        output_path: Path to save the rendered image (if None, the image is not saved)

    Returns:
        The rendered image as a numpy array
    """
    # Use the new render_mesh_with_vectors function
    return render_mesh_with_vectors(
        stl_mesh=stl_mesh,
        rotations=[rotation] if rotation is not None else None,
        image_size=image_size,
        view_elev=view_elev,
        view_azim=view_azim,
        output_path=output_path,
        show=show,
        show_vec=False
    )
