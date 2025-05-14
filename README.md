
# Darts Dataset Generator

This repository contains tools for generating datasets of rendered 3D models with random rotations.

## Installation

1. Install the basic dependencies:

```bash
pip install -r requirements.txt
```

2. For PyTorch3D, you need to install it from source:

```bash
# Install dependencies
pip install fvcore iopath

# For CUDA support (if you have a compatible GPU)
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# For CPU only or Apple Silicon
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Note: The PyTorch3D installation can be tricky. If you encounter issues, please refer to the [official installation guide](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

## Generate Renderings

### Using Blender (original method)

```bash
/Applications/Blender.app/Contents/MacOS/Blender --background --python scripts/generate_training_dataset.py -- --stl_path ./models/Dartfigt.stl --output_dir ./dart_dataset --num_images 1000 --image_res 256
```

### Using PyTorch3D (advanced method)

```bash
python scripts/generate_pytorch3d_dataset.py --stl_path ./models/Dartfigt.stl --output_dir ./pytorch3d_dataset --num_samples 1000 --image_size 256
```

The PyTorch3D method is faster and doesn't require Blender, but requires PyTorch3D to be installed correctly.

### Using Simple Renderer (fallback method)

If you have trouble installing PyTorch3D, you can use the simplified renderer:

```bash
# For NumPy < 2.0
python scripts/generate_simple_dataset.py --stl_path ./models/Dartfigt.stl --output_dir ./simple_dataset --num_samples 1000 --image_size 256

# For NumPy 2.0+
python scripts/generate_simple_dataset_numpy2.py --stl_path ./models/Dartfigt.stl --output_dir ./simple_dataset --num_samples 1000 --image_size 256
```

These methods use only basic libraries (NumPy, Matplotlib, scipy) and don't require PyTorch3D or Blender.

#### Additional Dependencies for Simple Renderer

```bash
# For the NumPy 2.0+ compatible version
pip install numpy-stl
```

## Dataset Format

Each sample is stored in a separate directory under the output directory:

```
output_dir/
  0000/
    image.png       # Rendered image
    rotation.txt    # Rotation parameters (Euler angles in radians)
  0001/
    image.png
    rotation.txt
  ...
```

## Visualizing Rotations

The repository includes utilities for visualizing rotations, which is useful for debugging and evaluating predicted rotations.

### Visualize a Single Rotation

```bash
# Visualize a specific rotation (in radians)
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 0.5,1.2,0.8

# Visualize a specific rotation (in degrees)
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 30,45,60 --degrees

# Customize the rotation vector appearance
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 0.5,1.2,0.8 --vec_scale 1.0 --true_color green
```

### Visualize True and Predicted Rotations

```bash
# Visualize true and predicted rotations (in radians)
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 0.5,1.2,0.8 --pred_rotation 0.6,1.1,0.9

# Visualize true and predicted rotations (in degrees)
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 30,45,60 --pred_rotation 35,40,65 --degrees

# Customize the rotation vector appearance
python scripts/visualize_rotation_simple.py single --stl_path ./models/Dartfigt.stl --true_rotation 0.5,1.2,0.8 --pred_rotation 0.6,1.1,0.9 --true_color green --pred_color purple --vec_scale 1.0
```

### Visualize a Dataset Sample

```bash
# Visualize a sample from the dataset
python scripts/visualize_rotation_simple.py dataset --stl_path ./models/Dartfigt.stl --dataset_dir ./simple_dataset --sample_idx 0

# Visualize a sample with a predicted rotation
python scripts/visualize_rotation_simple.py dataset --stl_path ./models/Dartfigt.stl --dataset_dir ./simple_dataset --sample_idx 0 --pred_rotation 0.6,1.1,0.9

# Customize the rotation vector appearance
python scripts/visualize_rotation_simple.py dataset --stl_path ./models/Dartfigt.stl --dataset_dir ./simple_dataset --sample_idx 0 --pred_rotation 0.6,1.1,0.9 --true_color green --pred_color purple --vec_scale 1.0
```
