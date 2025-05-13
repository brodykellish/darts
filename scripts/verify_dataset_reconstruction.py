import bpy
import os
import sys

dir = os.path.dirname(bpy.data.filepath)
if not dir in sys.path:
    sys.path.append(dir)

import numpy as np
import json
from mathutils import Vector, Euler
import argparse
from src.render.blender_utils import (
    clear_scene, setup_scene, load_stl,
    set_camera, apply_rotation, apply_gray_material, add_basic_light,
    render_to_file, add_rotation_vector, create_debug_grid
)

def main():
    parser = argparse.ArgumentParser(description='Verify dataset by rendering reconstructions.')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to STL file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with per-example subdirectories')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save reconstruction images')
    parser.add_argument('--image_res', type=int, default=256, help='Image resolution')
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    example_dirs = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    for exdir in example_dirs:
        ex_path = os.path.join(args.data_dir, exdir)
        rot_path = os.path.join(ex_path, 'rotation.txt')
        cam_path = os.path.join(ex_path, 'camera.json')
        if not (os.path.exists(rot_path) and os.path.exists(cam_path)):
            continue
        # Load camera and rotation
        with open(cam_path, 'r') as f:
            cam_params = json.load(f)
        rot = np.loadtxt(rot_path)
        print(f"\nProcessing example {exdir}")
        print(f"Rotation: {rot}")
        # Clear and setup scene
        clear_scene()
        setup_scene(args.image_res)
        
        # Load and prep model
        model = load_stl(args.stl_path)
        apply_rotation(model, rot)
        apply_gray_material(model)
        # Add rotation vector visualization
        rot_vec_obj = add_rotation_vector(model, rot)
        # Add debug grid
        debug_objects = create_debug_grid()
        # Camera and light
        camera = set_camera(cam_params)
        light = add_basic_light()
        
        # Print camera debug info
        print(f"Camera location: {camera.location}")
        print(f"Camera rotation: {camera.rotation_euler}")
        print(f"Camera lens: {camera.data.lens}")
        print(f"Camera clip start: {camera.data.clip_start}")
        print(f"Camera clip end: {camera.data.clip_end}")
        
        # Calculate and print camera frustum info
        cam_matrix = camera.matrix_world
        cam_direction = cam_matrix.to_3x3() @ Vector((0, 0, -1))
        print(f"Camera direction: {cam_direction}")
        print(f"Camera to origin distance: {(camera.location - Vector((0,0,0))).length}")
        
        # Render reconstruction
        out_ex_dir = os.path.join(args.output_dir, exdir)
        os.makedirs(out_ex_dir, exist_ok=True)
        recon_path = os.path.join(out_ex_dir, 'recon.png')
        render_to_file(recon_path)
        print(f"Saved: {recon_path}")
        
        # Cleanup
        bpy.data.objects.remove(light)
        bpy.data.objects.remove(rot_vec_obj)
        for obj in debug_objects:
            bpy.data.objects.remove(obj)

if __name__ == "__main__":
    main() 