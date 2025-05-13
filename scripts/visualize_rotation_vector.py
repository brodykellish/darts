import os
import numpy as np
from PIL import Image, ImageDraw
import argparse
import math
import json
from scipy.spatial.transform import Rotation as SciRot

def euler_to_matrix_xyz(euler):
    # Use scipy to match Blender's conventions exactly
    return SciRot.from_euler('xyz', euler).as_matrix()

def world_to_camera(vec, cam_loc, cam_rot_euler):
    R = euler_to_matrix_xyz(cam_rot_euler)
    vec_cam = R.T @ (np.array(vec) - np.array(cam_loc))
    return vec_cam

def project_vector_to_image(vec3, image_size, cam_loc, cam_rot_euler, fov_deg):
    X, Y, Z = world_to_camera(vec3, cam_loc, cam_rot_euler)
    fov_rad = np.radians(fov_deg)
    focal_length = (image_size / 2) / np.tan(fov_rad / 2)
    if Z <= 0:
        return None
    u = image_size / 2 + focal_length * X / Z
    v = image_size / 2 - focal_length * Y / Z
    return (int(u), int(v))

def clip_to_image(pt, image_res):
    x, y = pt
    x = max(0, min(image_res - 1, x))
    y = max(0, min(image_res - 1, y))
    return (x, y)

def main():
    parser = argparse.ArgumentParser(description='Overlay rotation vector on rendered images.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with images and rotation .txt files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save overlay images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(args.data_dir) if f.endswith('.png')])
    for fname in files:
        img_path = os.path.join(args.data_dir, fname)
        rot_path = os.path.join(args.data_dir, fname.replace('.png', '.txt'))
        cam_path = os.path.join(args.data_dir, fname.replace('.png', '_cam.json'))
        if not (os.path.exists(rot_path) and os.path.exists(cam_path)):
            continue
        # Load image, rotation, and camera params
        img = Image.open(img_path).convert('RGBA')
        draw = ImageDraw.Draw(img)
        rot = np.loadtxt(rot_path)
        with open(cam_path, 'r') as f:
            cam_params = json.load(f)
        cam_loc = cam_params['location']
        cam_rot_euler = cam_params['rotation_euler']
        fov_deg = cam_params['fov_deg']
        image_res = cam_params['image_res']
        cam_dist = np.linalg.norm(cam_loc)
        if rot.shape == (3,):
            # Euler angles (XYZ order)
            R = euler_to_matrix_xyz(rot)
        else:
            R = rot
        # The model's +Z axis in world coordinates
        z_axis = R @ np.array([0, 0, 1])
        arrow_length = cam_dist * 1.5  # Scale arrow by camera distance
        z_axis_vis = z_axis * arrow_length
        # Print camera info
        print(f"{fname}: cam_loc={cam_loc}, cam_rot_euler={cam_rot_euler}, fov_deg={fov_deg}, cam_dist={cam_dist}")
        print(f"z_axis_vis (world): {z_axis_vis}")
        z_axis_cam = world_to_camera(z_axis_vis, cam_loc, cam_rot_euler)
        print(f"z_axis_vis (camera): {z_axis_cam}")
        # Project the vector
        start = project_vector_to_image([0, 0, 0], image_res, cam_loc, cam_rot_euler, fov_deg)
        end = project_vector_to_image(z_axis_vis, image_res, cam_loc, cam_rot_euler, fov_deg)
        print(f"start={start}, end={end}")
        # Draw overlays
        if start and end:
            start = clip_to_image(start, image_res)
            end = clip_to_image(end, image_res)
            draw.line([start, end], fill=(255, 0, 0, 255), width=3)
            draw.ellipse([end[0]-4, end[1]-4, end[0]+4, end[1]+4], fill=(255,0,0,255))
        out_path = os.path.join(args.output_dir, fname)
        img.save(out_path)
        print(f"Saved overlay: {out_path}")

if __name__ == '__main__':
    main() 