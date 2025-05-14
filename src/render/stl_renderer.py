"""
Utility for rendering rotated STL models with an option to display a unit vector
indicating the direction of the rotation. This is useful for visually evaluating
predicted rotations at inference time.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
from PIL import Image
import os
from stl import mesh

def load_stl(file_path):
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
        sphere_radius = max_range * 0.05  # 5% of max range
        x = sphere_radius * np.cos(u) * np.sin(v)
        y = sphere_radius * np.sin(u) * np.sin(v)
        z = sphere_radius * np.cos(v)
        ax.plot_surface(x, y, z, color=color, alpha=0.8)

    # Plot the vector as an arrow with increased size and prominence
    quiver = ax.quiver(0, 0, 0, scaled_vec[0], scaled_vec[1], scaled_vec[2],
                      color=color, arrow_length_ratio=0.15, linewidth=3, label=label)

    return quiver

def render_mesh_with_vectors(stl_mesh, rotations=None, image_size=256,
                       view_elev=30, view_azim=45, output_path=None, show=True,
                       vec_colors=None, vec_labels=None, vec_scale=0.7, add_legend=True):
    """
    Render a mesh with multiple rotation vectors.

    Args:
        stl_mesh: A numpy-stl mesh object
        rotations: List of rotations (Rotation objects or Euler angles in radians)
        image_size: Size of the rendered image (square)
        view_elev: Elevation angle for the view
        view_azim: Azimuth angle for the view
        output_path: Path to save the rendered image (if None, the image is not saved)
        show: Whether to display the rendered image
        vec_colors: List of colors for the rotation vectors (default: auto-generated)
        vec_labels: List of labels for the rotation vectors (default: None)
        vec_scale: Scale factor for the rotation vectors as a fraction of the max range (default: 0.7)
        add_legend: Whether to add a legend for the vectors (default: True)

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

    # Create a Poly3DCollection
    mesh_collection = Poly3DCollection(triangles, alpha=0.8, edgecolor='k', linewidth=0.2)
    mesh_collection.set_facecolor('lightgray')
    ax.add_collection3d(mesh_collection)

    # Calculate the maximum range from the mesh vertices
    vertices = triangles.reshape(-1, 3)
    max_range = np.max(np.abs(vertices))

    # Add rotation vectors if provided
    if rotations:
        # Generate colors if not provided
        if vec_colors is None:
            # Use a colormap to generate distinct colors
            cmap = plt.cm.get_cmap('tab10', len(rotations))
            vec_colors = [cmap(i) for i in range(len(rotations))]
        elif isinstance(vec_colors, str):
            # If a single color is provided, use it for all vectors
            vec_colors = [vec_colors] * len(rotations)

        # Generate labels if not provided
        if vec_labels is None:
            vec_labels = [f"Rotation {i+1}" for i in range(len(rotations))]
        elif isinstance(vec_labels, str):
            # If a single label is provided, use it for all vectors
            vec_labels = [vec_labels] * len(rotations)

        # Add each rotation vector
        quivers = []
        for i, rotation in enumerate(rotations):
            quiver = add_rotation_vector(
                ax, rotation, max_range,
                color=vec_colors[i],
                scale=vec_scale,
                label=vec_labels[i],
                add_sphere=(i == 0)  # Only add sphere for the first vector
            )
            quivers.append(quiver)

        # Add a legend if requested and there are multiple vectors
        if add_legend and len(rotations) > 1:
            ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Set axis limits based on the mesh size
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)

    # Set view angle
    ax.view_init(elev=view_elev, azim=view_azim)

    # Remove axes
    ax.set_axis_off()

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

def render_mesh(stl_mesh, rotation=None, show_vec=False, image_size=256,
                view_elev=30, view_azim=45, output_path=None, show=True,
                vec_color='red', vec_scale=0.7):
    """
    Render a mesh with optional rotation and direction vector.

    Args:
        stl_mesh: A numpy-stl mesh object
        rotation: A scipy.spatial.transform.Rotation object or Euler angles in radians
        show_vec: Whether to display a unit vector indicating the direction of the rotation
        image_size: Size of the rendered image (square)
        view_elev: Elevation angle for the view
        view_azim: Azimuth angle for the view
        output_path: Path to save the rendered image (if None, the image is not saved)
        show: Whether to display the rendered image
        vec_color: Color of the rotation vector (default: 'red')
        vec_scale: Scale factor for the rotation vector as a fraction of the max range (default: 0.7)

    Returns:
        The rendered image as a numpy array
    """
    # Use the new render_mesh_with_vectors function
    if show_vec and rotation is not None:
        return render_mesh_with_vectors(
            stl_mesh=stl_mesh,
            rotations=[rotation],
            image_size=image_size,
            view_elev=view_elev,
            view_azim=view_azim,
            output_path=output_path,
            show=show,
            vec_colors=[vec_color],
            vec_labels=None,
            vec_scale=vec_scale,
            add_legend=False
        )
    else:
        return render_mesh_with_vectors(
            stl_mesh=stl_mesh,
            rotations=[rotation] if rotation is not None else None,
            image_size=image_size,
            view_elev=view_elev,
            view_azim=view_azim,
            output_path=output_path,
            show=show,
            add_legend=False
        )

def render_mesh_with_true_pred(stl_mesh, true_rotation, pred_rotation, image_size=256,
                           view_elev=30, view_azim=45, output_path=None, show=True,
                           true_color='red', pred_color='blue', vec_scale=0.7):
    """
    Render a mesh with both true and predicted rotation vectors on the same chart.

    Args:
        stl_mesh: A numpy-stl mesh object
        true_rotation: True rotation as Euler angles in radians or Rotation object
        pred_rotation: Predicted rotation as Euler angles in radians or Rotation object
        image_size: Size of the rendered image (square)
        view_elev: Elevation angle for the view
        view_azim: Azimuth angle for the view
        output_path: Path to save the rendered image (if None, the image is not saved)
        show: Whether to display the rendered image
        true_color: Color of the true rotation vector (default: 'red')
        pred_color: Color of the predicted rotation vector (default: 'blue')
        vec_scale: Scale factor for the rotation vectors as a fraction of the max range (default: 0.7)

    Returns:
        The rendered image as a numpy array
    """
    # Use the render_mesh_with_vectors function with both rotations
    return render_mesh_with_vectors(
        stl_mesh=stl_mesh,
        rotations=[true_rotation, pred_rotation],
        image_size=image_size,
        view_elev=view_elev,
        view_azim=view_azim,
        output_path=output_path,
        show=show,
        vec_colors=[true_color, pred_color],
        vec_labels=["True", "Predicted"],
        vec_scale=vec_scale,
        add_legend=True
    )
