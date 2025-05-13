import sys
import os
import bpy

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir )

import numpy as np
import random
import math
import argparse
import json
from src.render.blender_utils import (
    clear_scene, setup_scene, normalize_object, load_stl,
    create_centered_camera, add_random_light, apply_random_material, apply_random_rotation,
    render_image
)
import time
from mathutils import Vector, Euler

def print_progress(current, total, start_time):
    """Print a simple progress bar."""
    bar_length = 50
    filled_length = int(round(bar_length * current / float(total)))
    percents = round(100.0 * current / float(total), 1)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    
    # Calculate ETA
    elapsed_time = time.time() - start_time
    if current > 0:
        images_per_second = current / elapsed_time
        eta_seconds = (total - current) / images_per_second
        eta_str = f"ETA: {eta_seconds/60:.1f}min"
    else:
        eta_str = "ETA: calculating..."
    
    sys.stdout.write(f'\r[{bar}] {percents}% ({current}/{total}) {eta_str}')
    sys.stdout.flush()

def add_debug_helpers(model):
    # --- Draw bounding box ---
    bbox_corners = [model.matrix_world @ Vector(corner) for corner in model.bound_box]
    bbox_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # sides
    ]
    bbox_mesh = bpy.data.meshes.new("BBoxMesh")
    bbox_mesh.from_pydata(bbox_corners, bbox_edges, [])
    bbox_obj = bpy.data.objects.new("BBox", bbox_mesh)
    bpy.context.collection.objects.link(bbox_obj)
    bbox_obj.display_type = 'WIRE'
    bbox_obj.show_in_front = True
    # --- Draw origin point ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0))
    origin_sphere = bpy.context.active_object
    origin_mat = bpy.data.materials.new(name="OriginMat")
    origin_mat.use_nodes = True
    nodes = origin_mat.node_tree.nodes
    links = origin_mat.node_tree.links
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    # Add emission node for bright red
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    emission.inputs['Strength'].default_value = 10.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    origin_sphere.data.materials.append(origin_mat)
    # --- Draw rotation vector (model's +Z axis) ---
    rot_vec = model.matrix_world.to_3x3() @ Vector((0, 0, 1))
    arrow_length = 100
    arrow_end = rot_vec.normalized() * arrow_length
    arrow_mesh = bpy.data.meshes.new("RotVecMesh")
    arrow_mesh.from_pydata([(0, 0, 0), arrow_end], [(0, 1)], [])
    arrow_obj = bpy.data.objects.new("RotVec", arrow_mesh)
    bpy.context.collection.objects.link(arrow_obj)
    arrow_obj.display_type = 'WIRE'
    arrow_obj.show_in_front = True
    return [bbox_obj, origin_sphere, arrow_obj]

def remove_debug_helpers(helper_objs):
    for obj in helper_objs:
        bpy.data.objects.remove(obj)

def set_random_background():
    world = bpy.data.worlds[0]
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (
            random.random(),  # R
            random.random(),  # G
            random.random(),  # B
            1.0               # Alpha
        )
        bg.inputs[1].default_value = random.uniform(0.7, 1.5)  # Strength

def save_camera_params(camera, output_dir, index, image_res):
    params = {
        'location': list(camera.location),
        'rotation_euler': list(camera.rotation_euler),
        'fov_deg': math.degrees(camera.data.angle),
        'image_res': image_res
    }
    fname = f"{index:04d}_cam.json" if index is not None else "camera.json"
    with open(os.path.join(output_dir, fname), 'w') as f:
        json.dump(params, f, indent=2)

def cleanup_materials():
    """Remove all unused materials."""
    for material in bpy.data.materials:
        if not material.users:
            bpy.data.materials.remove(material)

def cleanup_meshes():
    """Remove all unused meshes."""
    for mesh in bpy.data.meshes:
        if not mesh.users:
            bpy.data.meshes.remove(mesh)

def cleanup_lights():
    """Remove all unused lights."""
    for light in bpy.data.lights:
        if not light.users:
            bpy.data.lights.remove(light)

def cleanup_scene():
    """Clear all objects and unused data from the scene."""
    # Remove all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Clean up unused data
    cleanup_materials()
    cleanup_meshes()
    cleanup_lights()
    
    # Clear world nodes
    world = bpy.data.worlds["World"]
    world.use_nodes = True
    world.node_tree.nodes.clear()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Render dataset from STL file')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for renders')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to render')
    parser.add_argument('--image_res', type=int, default=256, help='Image resolution')
    parser.add_argument('--debug', action='store_true', help='Include debug overlays (bounding box, origin, rotation vector) in every render')
    
    # Get arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    print(f"\nStarting dataset generation:")
    print(f"STL file: {args.stl_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of images: {args.num_images}")
    print(f"Image resolution: {args.image_res}x{args.image_res}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup scene
    print("Initializing scene...")
    cleanup_scene()
    setup_scene(args.image_res)
    model = load_stl(args.stl_path)
    model = normalize_object(model)  # Normalize only once after loading
    bpy.context.view_layer.objects.active = model
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    # Create a single fixed camera
    camera = create_centered_camera(model)

    # Render a preview of the normalized model at the origin (no rotation)
    print("Rendering preview of normalized model at the origin (no rotation)...")
    # Remove all materials
    model.data.materials.clear()
    # Add a simple gray material
    mat = bpy.data.materials.new(name="PreviewMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.0
        bsdf.inputs['Roughness'].default_value = 0.5
    model.data.materials.append(mat)
    # Add a white sun light
    light = bpy.data.lights.new(name="PreviewLight", type='SUN')
    light_obj = bpy.data.objects.new(name="PreviewLight", object_data=light)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (5, 5, 5)

    # --- Draw bounding box ---
    bbox_corners = [model.matrix_world @ Vector(corner) for corner in model.bound_box]
    bbox_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7)   # sides
    ]
    bbox_mesh = bpy.data.meshes.new("BBoxMesh")
    bbox_mesh.from_pydata(bbox_corners, bbox_edges, [])
    bbox_obj = bpy.data.objects.new("BBox", bbox_mesh)
    bpy.context.collection.objects.link(bbox_obj)
    bbox_obj.display_type = 'WIRE'
    bbox_obj.show_in_front = True
    # Set bounding box color (requires a material in Eevee, but in Cycles it will be white wire)

    # --- Draw origin point ---
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.03, location=(0, 0, 0))
    origin_sphere = bpy.context.active_object
    origin_mat = bpy.data.materials.new(name="OriginMat")
    origin_mat.use_nodes = True
    nodes = origin_mat.node_tree.nodes
    links = origin_mat.node_tree.links
    # Remove default nodes
    for node in nodes:
        nodes.remove(node)
    # Add emission node for bright red
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1.0, 0.0, 0.0, 1.0)
    emission.inputs['Strength'].default_value = 10.0
    output = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    origin_sphere.data.materials.append(origin_mat)

    # --- Draw rotation vector (model's +Z axis) ---
    rot_vec = model.matrix_world.to_3x3() @ Vector((0, 0, 1))
    arrow_length = 0.5
    arrow_end = rot_vec.normalized() * arrow_length
    # Create a mesh line for the arrow
    arrow_mesh = bpy.data.meshes.new("RotVecMesh")
    arrow_mesh.from_pydata([(0, 0, 0), arrow_end], [(0, 1)], [])
    arrow_obj = bpy.data.objects.new("RotVec", arrow_mesh)
    bpy.context.collection.objects.link(arrow_obj)
    arrow_obj.display_type = 'WIRE'
    arrow_obj.show_in_front = True

    # Render preview
    preview_path = os.path.join(args.output_dir, "origin_preview.png")
    bpy.context.scene.camera = camera
    bpy.context.scene.render.filepath = preview_path
    bpy.ops.render.render(write_still=True)
    print(f"Preview render saved to: {preview_path}\n")

    # Remove preview helpers
    bpy.data.objects.remove(light_obj)
    bpy.data.objects.remove(bbox_obj)
    bpy.data.objects.remove(origin_sphere)
    bpy.data.objects.remove(arrow_obj)
    model.data.materials.clear()
    cleanup_materials()
    cleanup_meshes()
    cleanup_lights()

    # Generate renders
    print("\nGenerating renders...")
    start_time = time.time()
    for i in range(args.num_images):
        # Create subdirectory for this example
        example_dir = os.path.join(args.output_dir, f"{i:04d}")
        os.makedirs(example_dir, exist_ok=True)
        # Apply random rotation and material (do NOT normalize again)
        apply_random_rotation(model)
        apply_random_material(model)
        # Add random lighting
        light = add_random_light()
        # Add debug overlays if requested
        debug_helpers = []
        if args.debug:
            debug_helpers = add_debug_helpers(model)
        # Set random background
        set_random_background()
        # Save camera parameters
        save_camera_params(camera, example_dir, None, args.image_res)  # None for index, will use 'camera.json'
        # Render and save
        render_image(None, camera, example_dir)  # None for index, will use 'image.png' and 'rotation.txt'
        # Cleanup
        bpy.data.objects.remove(light)
        if args.debug:
            remove_debug_helpers(debug_helpers)
        # Clean up unused data every 10 iterations
        if i % 10 == 0:
            cleanup_materials()
            cleanup_meshes()
            cleanup_lights()
        # Update progress
        print_progress(i + 1, args.num_images, start_time)
    
    print("\n")  # New line after progress bar
    total_time = time.time() - start_time
    print(f"\nDataset generation complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Average speed: {args.num_images/total_time:.1f} images/second")
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 