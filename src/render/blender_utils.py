import bpy
import os
import mathutils
import numpy as np
import random
import math
from mathutils import Vector, Euler

def clear_scene():
    """Clear all objects from the Blender scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_scene(resolution=256):
    """Configure the Blender scene for rendering."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Set up world background to black
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0, 0, 0, 1)  # Black background
    bg.inputs[1].default_value = 1.0  # Strength
    
    # Ensure proper rendering of emissive materials
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01
    scene.cycles.adaptive_min_samples = 16

def normalize_object(obj):
    """Normalize object to be centered at origin and have uniform size."""
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, Vector((0, 0, 0))) / len(bbox_corners)
    bbox_size = max((max(corner[i] for corner in bbox_corners) - 
                    min(corner[i] for corner in bbox_corners)) 
                   for i in range(3))
    target_size = 1.0
    scale_factor = target_size / bbox_size
    obj.location -= center
    obj.scale *= scale_factor
    obj.matrix_world.translation = obj.location
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    return obj

def load_stl(filepath):
    """Load and normalize an STL file."""
    # Try both old and new import operators
    try:
        bpy.ops.import_mesh.stl(filepath=filepath)
    except AttributeError:
        try:
            bpy.ops.wm.stl_import(filepath=filepath)
        except AttributeError:
            raise Exception("Could not find STL import operator. Please check your Blender version.")
    obj = bpy.context.active_object
    obj.name = "Model"
    return normalize_object(obj)

def get_bounding_sphere_radius(obj):
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return max((corner - Vector((0, 0, 0))).length for corner in bbox_corners)

def create_centered_camera(obj, fov_deg=50):
    scene = bpy.context.scene
    radius = get_bounding_sphere_radius(obj)
    fov_rad = math.radians(fov_deg)
    cam_dist = radius / math.sin(fov_rad / 2)
    location = (0, cam_dist * 1.5, 0)
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    direction = Vector((0, 0, 0)) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    cam.data.angle = fov_rad
    scene.camera = cam
    return cam

def add_random_light():
    """Add a random light source to the scene."""
    light_types = ['POINT', 'SUN', 'AREA']
    light_type = random.choice(light_types)
    bpy.ops.object.light_add(type=light_type)
    light = bpy.context.active_object
    
    # Position light randomly on a sphere
    theta = random.uniform(0, 2 * math.pi)
    phi = random.uniform(0, math.pi)
    radius = random.uniform(2.0, 4.0)
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    light.location = (x, y, z)
    
    # Randomize light properties
    light.data.energy = random.uniform(500, 2000)
    if light_type == 'AREA':
        light.data.size = random.uniform(1.0, 3.0)
    return light

def apply_random_material(obj):
    mat = bpy.data.materials.new(name="RandomMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    
    # Random material properties
    principled.inputs['Base Color'].default_value = (
        random.random(),
        random.random(),
        random.random(),
        1.0
    )
    principled.inputs['Metallic'].default_value = random.random()
    principled.inputs['Roughness'].default_value = random.random()
    
    # Connect nodes
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    print(f"[DEBUG] Material color: {mat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value}")
    return obj

def apply_random_rotation(obj):
    """Apply a random rotation to the object."""
    # Generate random Euler angles
    euler = Euler((
        random.uniform(0, 2 * math.pi),  # X rotation
        random.uniform(0, 2 * math.pi),  # Y rotation
        random.uniform(0, 2 * math.pi)   # Z rotation
    ))
    obj.rotation_euler = euler
    return obj

def render_image(index, camera, output_dir):
    """Render an image and save rotation data."""
    bpy.context.scene.camera = camera
    if index is not None:
        filepath = os.path.join(output_dir, f"{index:04d}.png")
        rotation_path = os.path.join(output_dir, f"{index:04d}.txt")
    else:
        filepath = os.path.join(output_dir, "image.png")
        rotation_path = os.path.join(output_dir, "rotation.txt")
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    
    # Save rotation data (Euler angles)
    model = bpy.data.objects["Model"]
    rotation = model.rotation_euler
    np.savetxt(rotation_path, np.array(rotation))

def set_camera(cam_params):
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.camera_add(location=cam_params['location'])
    cam = bpy.context.active_object
    cam.rotation_euler = Euler(cam_params['rotation_euler'])
    cam.data.angle = np.radians(cam_params['fov_deg'])
    bpy.context.scene.camera = cam
    return cam

def apply_rotation(obj, rot):
    obj.rotation_euler = Euler(rot)
    return obj

def apply_gray_material(obj):
    mat = bpy.data.materials.new(name="GrayMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.5
    mat.node_tree.links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    return obj

def add_basic_light():
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    return bpy.context.active_object

def render_to_file(filepath):
    bpy.context.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)

def add_rotation_vector(model, rotation_data):
    # Convert rotation data to Euler angles and create rotation matrix
    euler = Euler(rotation_data)
    rot_matrix = euler.to_matrix()
    
    # Get the transformed Z-axis (0,0,1) after rotation
    z_axis = Vector((0, 0, 1))
    transformed_z = rot_matrix @ z_axis
    
    tip_point = Vector((0, 0, 0))
    
    arrow_length = -100.0
    arrow_start = tip_point
    arrow_end = tip_point + transformed_z.normalized() * arrow_length
    
    # Create a curve for the arrow shaft
    curve_data = bpy.data.curves.new('RotVecCurve', type='CURVE')
    curve_data.dimensions = '3D'
    curve_data.resolution_u = 2
    
    # Create a polyline for the shaft
    polyline = curve_data.splines.new('POLY')
    polyline.points.add(1)  # Add one point for the end
    polyline.points[0].co = (*arrow_start, 1)  # Start point
    polyline.points[1].co = (*arrow_end, 1)  # End point
    
    # Create arrowhead points
    arrow_head_length = arrow_length * 0.1  # 10% of shaft length
    arrow_head_width = arrow_head_length * 0.5
    
    # Create arrowhead base point
    arrow_head_base = arrow_end - transformed_z.normalized() * arrow_head_length
    
    # Create perpendicular vectors for arrowhead
    perp1 = transformed_z.cross(Vector((1, 0, 0)))
    if perp1.length < 0.1:  # If transformed_z is parallel to X axis
        perp1 = transformed_z.cross(Vector((0, 1, 0)))
    perp1.normalize()
    perp2 = transformed_z.cross(perp1).normalized()
    
    # Create arrowhead points
    arrowhead_points = [
        arrow_head_base + perp1 * arrow_head_width,
        arrow_head_base - perp1 * arrow_head_width,
        arrow_head_base + perp2 * arrow_head_width,
        arrow_head_base - perp2 * arrow_head_width
    ]
    
    # Add arrowhead splines
    for point in arrowhead_points:
        spline = curve_data.splines.new('POLY')
        spline.points.add(1)
        spline.points[0].co = (*point, 1)
        spline.points[1].co = (*arrow_end, 1)
    
    # Create the curve object
    arrow_obj = bpy.data.objects.new("RotVec", curve_data)
    bpy.context.collection.objects.link(arrow_obj)
    
    # Set curve thickness
    curve_data.bevel_depth = 0.5  # Thickness of the line
    curve_data.bevel_resolution = 2  # Smoothness of the bevel
    
    # Create material for the arrow
    arrow_mat = bpy.data.materials.new(name="ArrowMat")
    arrow_mat.use_nodes = True
    nodes = arrow_mat.node_tree.nodes
    links = arrow_mat.node_tree.links
    
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add emission node for bright red
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)  # Bright red
    emission.inputs['Strength'].default_value = 50.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    arrow_obj.data.materials.append(arrow_mat)
    
    # Print debug information
    print(f"Original Z-axis: {z_axis}")
    print(f"Transformed Z-axis: {transformed_z}")
    print(f"Rotation angles (degrees): {[math.degrees(angle) for angle in rotation_data]}")
    print(f"Model tip point: {tip_point}")
    print(f"Arrow start: {arrow_start}")
    print(f"Arrow end: {arrow_end}")
    
    return arrow_obj
def create_debug_grid():
    # Create a collection for debug objects
    debug_collection = bpy.data.collections.new("DebugGrid")
    bpy.context.scene.collection.children.link(debug_collection)
    
    # Create material for debug lines
    debug_mat = bpy.data.materials.new(name="DebugMat")
    debug_mat.use_nodes = True
    nodes = debug_mat.node_tree.nodes
    links = debug_mat.node_tree.links
    
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add emission node for bright yellow
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 1.0, 0.0, 1.0)  # Bright yellow
    emission.inputs['Strength'].default_value = 10.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    debug_objects = []
    
    # Create X, Y, Z axes
    axes = [
        [(0, 0, 0), (10, 0, 0)],  # X axis
        [(0, 0, 0), (0, 10, 0)],  # Y axis
        [(0, 0, 0), (0, 0, 10)]   # Z axis
    ]
    
    for start, end in axes:
        # Create a line mesh
        mesh = bpy.data.meshes.new("DebugLine")
        mesh.from_pydata([start, end], [(0, 1)], [])
        line = bpy.data.objects.new("DebugLine", mesh)
        line.data.materials.append(debug_mat)
        line.show_in_front = True
        debug_collection.objects.link(line)
        debug_objects.append(line)
    
    # Create a grid in the XY plane
    grid_size = 10
    grid_step = 1
    for x in range(-grid_size, grid_size + 1, grid_step):
        # X lines
        mesh = bpy.data.meshes.new("GridLine")
        mesh.from_pydata([(x, -grid_size, 0), (x, grid_size, 0)], [(0, 1)], [])
        line = bpy.data.objects.new("GridLine", mesh)
        line.data.materials.append(debug_mat)
        line.show_in_front = True
        debug_collection.objects.link(line)
        debug_objects.append(line)
    
    for y in range(-grid_size, grid_size + 1, grid_step):
        # Y lines
        mesh = bpy.data.meshes.new("GridLine")
        mesh.from_pydata([(-grid_size, y, 0), (grid_size, y, 0)], [(0, 1)], [])
        line = bpy.data.objects.new("GridLine", mesh)
        line.data.materials.append(debug_mat)
        line.show_in_front = True
        debug_collection.objects.link(line)
        debug_objects.append(line)
    
    return debug_objects

def create_test_cube():
    # Create a simple cube at the origin
    bpy.ops.mesh.primitive_cube_add(size=50.0, location=(0, 0, 0))
    cube = bpy.context.active_object
    
    # Create bright red emissive material
    mat = bpy.data.materials.new(name="TestCubeMat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    
    # Add emission node for bright red
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)  # Bright red
    emission.inputs['Strength'].default_value = 50.0  # Increased strength significantly
    
    # Add a mix shader to combine emission with a basic material
    mix = nodes.new(type='ShaderNodeMixShader')
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.inputs['Base Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.0
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    
    # Connect nodes
    links.new(emission.outputs['Emission'], mix.inputs[1])
    links.new(principled.outputs['BSDF'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output.inputs['Surface'])
    
    # Set mix factor to 0.5 to combine emission and material
    mix.inputs[0].default_value = 0.5
    
    cube.data.materials.append(mat)
    
    # Make sure the cube is visible in render
    cube.hide_render = False
    cube.hide_viewport = False
    
    return cube 