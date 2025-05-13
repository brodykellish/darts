# Dart Rendering Pipeline Codebook

## Overview
This pipeline generates synthetic training data for dart pose estimation by rendering a 3D dart model from various viewpoints with randomized lighting and materials.

## Dependencies
- Blender 4.4.3 or later
- Python 3.11
- NumPy
- Blender Python API (bpy)

## Command Line Arguments
```bash
python render_dart.py --obj_path <path_to_stl> --output_dir <output_directory> [--num_images <N>] [--image_res <size>]
```

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--obj_path` | str | Yes | - | Path to the dart STL file |
| `--output_dir` | str | Yes | - | Directory to save rendered images and poses |
| `--num_images` | int | No | 1000 | Number of images to render |
| `--image_res` | int | No | 256 | Resolution of rendered images (square) |

## Output Files
For each rendered image, two files are generated:
1. `{index:04d}.png` - Rendered image
2. `{index:04d}.txt` - Camera pose matrix (3x4) containing rotation (3x3) and translation (3x1)

## Key Functions

### `load_dart(obj_path)`
Imports and prepares the dart model:
- Imports STL file
- Names object "Dart"
- Scales to 0.01 units
- Centers at origin
- Applies smooth shading

### `random_camera_pose()`
Generates random camera positions:
- Spherical coordinates around origin
- Radius: 1.3-1.8 units
- Full spherical coverage

### `create_camera(point, target)`
Creates and configures camera:
- Position: specified point
- Target: specified point (defaults to origin)
- Orientation: -Z forward, Y up

### `add_random_light()`
Adds randomized lighting:
- Position: random spherical coordinates
- Energy: 500-2000 units
- Color: random RGB

### `apply_random_material(obj)`
Applies randomized material:
- Base color: random RGB
- Roughness: 0.2-0.9

## Scene Configuration
- Render engine: Cycles
- Device: GPU
- Background: Random color
- Film: Non-transparent

## Camera Parameters
- Position: Random spherical coordinates
- Target: Dart object location
- Distance: 1.3-1.8 units from origin

## Lighting Parameters
- Type: Point light
- Position: Random spherical coordinates
- Energy: 500-2000 units
- Color: Random RGB

## Material Parameters
- Type: Principled BSDF
- Base color: Random RGB
- Roughness: 0.2-0.9

## Usage Example
```bash
python render_dart.py --obj_path ./models/Dart_standart.obj --output_dir ./models --num_images 1000 --image_res 256
```

## Debug Logging
The script includes extensive debug logging for:
- Argument parsing
- Scene setup
- Object import and transformation
- Camera creation and positioning
- Light creation and parameters
- Material application
- Rendering process

## Notes
- The dart model is scaled to 0.01 units to match typical Blender units
- Camera positions are generated in a spherical pattern around the dart
- Each render includes a single random point light
- The background color is randomized for each render
- All random parameters are seeded with the system time 