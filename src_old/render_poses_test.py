import bpy
import json
import math
import mathutils
import os
import argparse
import numpy as np
from src.blender_utils import (
    clear_scene, setup_scene, load_stl, create_camera,
    add_light, apply_fixed_material, apply_pose, render_image
)

def parse_args():
    parser = argparse.ArgumentParser(description='Render STL with predicted poses')
    parser.add_argument('--stl_path', type=str, required=True, help='Path to the STL file for rendering')
    parser.add_argument('--eval_dir', type=str, required=True, help='Path to evaluation directory containing test cases')
    parser.add_argument('--output_dir', type=str, default='renders', help='Directory to save renders')
    parser.add_argument('--image_res', type=int, default=256, help='Image resolution (square)')
    args, _ = parser.parse_known_args()
    print(f"[DEBUG] Parsed args: {args}")
    return args

def setup_camera():
    print("[DEBUG] Setting up fixed camera...")
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.location = (0, -2, 0)
    cam.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = cam
    print(f"[DEBUG] Camera location: {cam.location}, rotation: {cam.rotation_euler}")
    return cam

def main():
    args = parse_args()
    print(f"[DEBUG] Starting main with args: {args}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[DEBUG] Created output directory: {args.output_dir}")
    
    # Setup scene
    clear_scene()
    setup_scene(args.image_res)
    
    # Load and prepare STL
    dart = load_stl(args.stl_path)
    apply_fixed_material(dart)
    
    # Setup camera and lighting
    setup_camera()
    add_light()
    
    # Process each test case
    test_cases = sorted([d for d in os.listdir(args.eval_dir) if d.startswith('test_case_')])
    print(f"[DEBUG] Found test cases: {test_cases}")
    
    for case_dir in test_cases:
        print(f"[DEBUG] Processing test case: {case_dir}")
        case_path = os.path.join(args.eval_dir, case_dir)
        
        # Load poses
        pose_gt = np.loadtxt(os.path.join(case_path, 'pose_gt.txt'))
        pose_pred = np.loadtxt(os.path.join(case_path, 'pose_pred.txt'))
        print(f"[DEBUG] Loaded poses for {case_dir}")
        
        # Extract rotation and translation
        R_gt = pose_gt[:3, :3]
        t_gt = pose_gt[:3, 3]
        R_pred = pose_pred[:3, :3]
        t_pred = pose_pred[:3, 3]
        
        # Render ground truth
        apply_pose(dart, R_gt, t_gt)
        gt_path = os.path.join(args.output_dir, f'{case_dir}_gt.png')
        bpy.context.scene.render.filepath = gt_path
        bpy.ops.render.render(write_still=True)
        
        # Render prediction
        apply_pose(dart, R_pred, t_pred)
        pred_path = os.path.join(args.output_dir, f'{case_dir}_pred.png')
        bpy.context.scene.render.filepath = pred_path
        bpy.ops.render.render(write_still=True)
        
        print(f"[DEBUG] Completed rendering for {case_dir}")

if __name__ == "__main__":
    main() 