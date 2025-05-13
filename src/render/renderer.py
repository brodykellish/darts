import os
import bpy
from . import blender_utils

class ModelRenderer:
    """High-level interface for rendering 3D models."""
    
    def __init__(self, stl_path, output_dir, image_res=256):
        """Initialize the renderer with model and output settings."""
        self.stl_path = stl_path
        self.output_dir = output_dir
        self.image_res = image_res
        os.makedirs(output_dir, exist_ok=True)
        
    def setup_scene(self):
        """Set up the Blender scene for rendering."""
        blender_utils.clear_scene()
        blender_utils.setup_scene(self.image_res)
        self.model = blender_utils.load_stl(self.stl_path)
        self.camera = blender_utils.create_fixed_camera()
        
    def render_dataset(self, num_images):
        """Generate a dataset of rendered images and rotations."""
        self.setup_scene()
        
        for i in range(num_images):
            # Apply random rotation and material
            blender_utils.apply_random_rotation(self.model)
            blender_utils.apply_random_material(self.model)
            
            # Add random lighting
            light = blender_utils.add_random_light()
            
            # Render and save
            blender_utils.render_image(i, self.camera, self.output_dir)
            
            # Cleanup
            bpy.data.objects.remove(light)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_images} samples")
                
    def render_rotation(self, rotation, output_path):
        """Render the model in a specific rotation."""
        self.setup_scene()
        
        # Apply specific rotation
        self.model.rotation_euler = rotation
        blender_utils.apply_random_material(self.model)
        
        # Add random lighting
        light = blender_utils.add_random_light()
        
        # Render
        bpy.context.scene.camera = self.camera
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        
        # Cleanup
        bpy.data.objects.remove(light) 