import bpy
import os
import argparse
from blender_utils import (
    clear_scene, setup_scene, load_stl, random_camera_pose,
    create_camera, add_random_light, apply_random_material,
    render_image
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, required=True, help='Path to the dart STL file')
    parser.add_argument('--output_dir', type=str, default='data', help='Where to save rendered images and poses')
    parser.add_argument('--num_images', type=int, default=500, help='Number of images to render')
    parser.add_argument('--image_res', type=int, default=256, help='Image resolution (square)')
    args, _ = parser.parse_known_args()
    print(f"[DEBUG] Parsed args: {args}")
    return args

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup scene and load model
    clear_scene()
    setup_scene(args.image_res)
    dart = load_stl(args.obj_path)
    apply_random_material(dart)

    # Generate training samples
    for i in range(args.num_images):
        # Setup camera and lighting
        cam_pos = random_camera_pose()
        cam = create_camera(cam_pos, target=(0, 0, 0))  # Target is origin since model is normalized
        light = add_random_light()
        
        # Render and save
        render_image(i, cam, args.output_dir)
        
        # Cleanup for next iteration
        bpy.data.objects.remove(cam)
        bpy.data.objects.remove(light)
        
        if (i + 1) % 50 == 0:
            print(f"Generated {i + 1}/{args.num_images} samples")

if __name__ == "__main__":
    main()
