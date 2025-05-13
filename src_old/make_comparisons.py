import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse

def make_comparisons(render_dir, eval_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    gt_imgs = [f for f in os.listdir(render_dir) if f.endswith('_gt.png')]
    for gt_img in gt_imgs:
        case_id = gt_img[:-7]  # Remove '_gt.png'
        pred_img = f'{case_id}_pred.png'
        if not os.path.exists(os.path.join(render_dir, pred_img)):
            continue
        gt_path = os.path.join(render_dir, gt_img)
        pred_path = os.path.join(render_dir, pred_img)
        # Find the corresponding input image in the evaluation directory
        # Assume test_case_xxx/input.png
        test_case_dir = os.path.join(eval_dir, case_id)
        input_path = os.path.join(test_case_dir, 'input.png')
        if not os.path.exists(input_path):
            print(f"Warning: input image not found for {case_id}")
            continue
        # Load images
        input_img = Image.open(input_path)
        gt = Image.open(gt_path)
        pred = Image.open(pred_path)
        # Create side-by-side (Input | GT | Prediction)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(input_img)
        axes[0].set_title('Input')
        axes[0].axis('off')
        axes[1].imshow(gt)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        axes[2].imshow(pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        plt.tight_layout()
        comp_path = os.path.join(output_dir, f'{case_id}_comparison.png')
        plt.savefig(comp_path)
        plt.close()
        print(f"Created {comp_path}")

def main():
    parser = argparse.ArgumentParser(description='Create side-by-side comparison images from renders')
    parser.add_argument('--render-dir', type=str, required=True, help='Directory with rendered images')
    parser.add_argument('--eval-dir', type=str, required=True, help='Directory with evaluation test cases (input images)')
    parser.add_argument('--output-dir', type=str, default='comparisons', help='Directory to save comparisons')
    args = parser.parse_args()
    make_comparisons(args.render_dir, args.eval_dir, args.output_dir)

if __name__ == "__main__":
    main() 