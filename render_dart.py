import bpy
import os
import mathutils
import numpy as np
import random
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_path', type=str, required=True, help='Path to the dart STL file')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to save rendered images and poses')
    parser.add_argument('--num_images', type=int, default=1000, help='Number of images to render')
    parser.add_argument('--image_res', type=int, default=256, help='Image resolution (square)')
    args, _ = parser.parse_known_args(sys.argv[sys.argv.index('--') + 1:])
    print(f"[DEBUG] Parsed args: {args}")
    return args

def clear_scene():
    print("[DEBUG] Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_scene(image_res):
    print(f"[DEBUG] Setting up scene with image_res={image_res}")
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.film_transparent = False
    scene.render.resolution_x = image_res
    scene.render.resolution_y = image_res

    # Random world background color
    bg_color = (random.random(), random.random(), random.random(), 1)
    bg = bpy.data.worlds["World"].node_tree.nodes["Background"]
    bg.inputs[0].default_value = bg_color
    bg.inputs[1].default_value = 1.0
    print(f"[DEBUG] Set random background color: {bg_color}")

def load_dart(obj_path):
    print(f"[DEBUG] Importing STL from: {obj_path}")
    bpy.ops.wm.stl_import(filepath=obj_path)
    obj = bpy.context.selected_objects[0]
    print(f"[DEBUG] Imported object: {obj.name}")
    obj.name = "Dart"
    obj.scale = (0.01, 0.01, 0.01)
    bpy.context.view_layer.update()
    print(f"[DEBUG] Scaled object to: {obj.scale}")
    # Move object to origin
    obj.location = (0, 0, 0)
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    bpy.context.view_layer.update()
    print(f"[DEBUG] Object location after centering: {obj.location}")
    bpy.ops.object.shade_smooth()
    print(f"[DEBUG] Finished preparing object: {obj.name}")
    return obj

def random_camera_pose():
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    r = np.random.uniform(1.3, 1.8)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * costheta
    print(f"[DEBUG] Generated random camera position: ({x}, {y}, {z})")
    return (x, y, z)

def create_camera(point, target=(0, 0, 0)):
    print(f"[DEBUG] Creating camera at {point}, targeting {target}")
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.location = point
    direction = mathutils.Vector(target) - cam.location
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
    light_data = bpy.data.lights.new(name="light", type='POINT')
    light_obj = bpy.data.objects.new(name="light", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    theta = random.uniform(0, 2*np.pi)
    phi = random.uniform(0, np.pi)
    r = random.uniform(1.5, 3.0)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    light_obj.location = (x, y, abs(z))
    print(f"[DEBUG] Light position: {light_obj.location}")

    light_obj.data.energy = random.uniform(500, 2000)
    light_obj.data.color = (random.random(), random.random(), random.random())
    print(f"[DEBUG] Light energy: {light_obj.data.energy}, color: {light_obj.data.color}")
    return light_obj

def apply_random_material(obj):
    print(f"[DEBUG] Applying random material to: {obj.name}")
    mat = bpy.data.materials.new(name="DartMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    color = (random.random(), random.random(), random.random(), 1.0)
    bsdf.inputs["Base Color"].default_value = color
    roughness = random.uniform(0.2, 0.9)
    bsdf.inputs["Roughness"].default_value = roughness
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    print(f"[DEBUG] Material color: {color}, roughness: {roughness}")

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

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    clear_scene()
    setup_scene(args.image_res)

    dart = load_dart(args.obj_path)
    apply_random_material(dart)
    print(f"[DEBUG] Dart object location before rendering: {dart.location}, scale: {dart.scale}")
    for i in range(args.num_images):
        cam_pos = random_camera_pose()
        cam = create_camera(cam_pos, target=tuple(dart.location))
        light = add_random_light()
        print(f"[DEBUG] Rendering image {i} with camera at {cam.location}, looking at {dart.location}")
        render_image(i, cam, args.output_dir)
        bpy.data.objects.remove(cam)
        bpy.data.objects.remove(light)

if __name__ == "__main__":
    main()
