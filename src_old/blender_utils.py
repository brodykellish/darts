import bpy
import os
import mathutils
import numpy as np
import random
import math
from mathutils import Vector

def clear_scene():
    print("[DEBUG] Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_scene(resolution=256):
    print(f"[DEBUG] Setting up scene with resolution={resolution}")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Random world background color
    bg_color = (random.random(), random.random(), random.random(), 1)
    bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
    bg.inputs[0].default_value = bg_color
    bg.inputs[1].default_value = 1.0
    print(f"[DEBUG] Set random background color: {bg_color}")

def normalize_object(obj):
    """Normalize object to be centered at origin and have uniform size."""
    # Get the object's bounding box
    bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    # Calculate the center of the bounding box
    center = sum(bbox_corners, Vector((0, 0, 0))) / len(bbox_corners)
    
    # Calculate the size of the bounding box
    bbox_size = max((max(corner[i] for corner in bbox_corners) - 
                    min(corner[i] for corner in bbox_corners)) 
                   for i in range(3))
    
    # Calculate scale factor to normalize size (target size = 1.0 units)
    target_size = 1.0
    scale_factor = target_size / bbox_size
    
    # Apply transformations
    obj.location -= center  # Center at origin
    obj.scale *= scale_factor  # Normalize size
    
    # Update the object's matrix
    obj.matrix_world.translation = obj.location
    
    print(f"[DEBUG] Normalized object: location={obj.location}, scale={obj.scale}")
    return obj

def load_stl(filepath):
    print(f"[DEBUG] Importing STL from: {filepath}")
    bpy.ops.import_mesh.stl(filepath=filepath)
    obj = bpy.context.active_object
    print(f"[DEBUG] Imported object: {obj.name}")
    obj.name = "Dart"
    obj = normalize_object(obj)
    bpy.context.view_layer.update()
    print(f"[DEBUG] Finished preparing object: {obj.name}")
    return obj

def random_camera_pose():
    # Generate random camera position on a sphere
    radius = 2.0  # Distance from origin
    theta = random.uniform(0, 2 * math.pi)  # Azimuth
    phi = random.uniform(0, math.pi)  # Polar angle
    
    x = radius * math.sin(phi) * math.cos(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(phi)
    
    print(f"[DEBUG] Generated random camera position: ({x}, {y}, {z})")
    return (x, y, z)

def create_camera(location, target=(0, 0, 0)):
    print(f"[DEBUG] Creating camera at {location}, targeting {target}")
    bpy.ops.object.camera_add(location=location)
    cam = bpy.context.active_object
    
    # Point camera at target
    direction = Vector(target) - Vector(location)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    print(f"[DEBUG] Camera rotation_euler: {cam.rotation_euler}")
    return cam

def add_light():
    print("[DEBUG] Adding fixed light at (2, 2, 2)")
    light_data = bpy.data.lights.new(name="light", type='POINT')
    light_obj = bpy.data.objects.new(name="light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (2, 2, 2)
    light_obj.data.energy = 1000
    return light_obj

def add_random_light():
    print("[DEBUG] Adding random light...")
    light_types = ['POINT', 'SUN', 'AREA']
    light_type = random.choice(light_types)
    
    bpy.ops.object.light_add(type=light_type)
    light = bpy.context.active_object
    
    # Random position
    light.location = random_camera_pose()
    
    # Random energy
    light.data.energy = random.uniform(500, 2000)
    
    return light

def apply_random_material(obj):
    print(f"[DEBUG] Applying random material to: {obj.name}")
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

def apply_fixed_material(obj):
    print(f"[DEBUG] Applying fixed material to: {obj.name}")
    mat = bpy.data.materials.new(name="DartMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = (0.7, 0.2, 0.2, 1.0)  # Fixed reddish color
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.5
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print(f"[DEBUG] Material color: {color}, roughness: 0.5")

def render_image(index, camera, output_dir):
    print(f"[DEBUG] Rendering image {index}...")
    bpy.context.scene.camera = camera
    filepath = os.path.join(output_dir, f"{index:04d}.png")
    bpy.context.scene.render.filepath = filepath
    print(f"[DEBUG] Output filepath: {filepath}")
    bpy.ops.render.render(write_still=True)

    R = np.array(camera.matrix_world.to_3x3().transposed())
    t = np.array(camera.location)
    pose_path = os.path.join(output_dir, f"{index:04d}.txt")
    np.savetxt(pose_path, np.hstack((R, t.reshape(3, 1))))
    print(f"[DEBUG] Saved pose to: {pose_path}")

def apply_pose(obj, R, t):
    print(f"[DEBUG] Applying pose to {obj.name}")
    print(f"[DEBUG] Rotation matrix:\n{R}")
    print(f"[DEBUG] Translation vector: {t}")
    
    # Convert numpy arrays to mathutils types
    R_blender = mathutils.Matrix(R.tolist())
    t_blender = mathutils.Vector(t.tolist())
    
    # Apply rotation
    obj.rotation_euler = R_blender.to_euler()
    print(f"[DEBUG] Applied rotation: {obj.rotation_euler}")
    
    # Apply translation
    obj.location = t_blender
    print(f"[DEBUG] Applied translation: {obj.location}")
    
    # Update the scene
    bpy.context.view_layer.update() 