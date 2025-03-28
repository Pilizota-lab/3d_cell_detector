import os
import re
import numpy as np
import matplotlib.pyplot as plt
from cellpose import io, plot
from pathlib import Path

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def convert_seg_to_label(seg_file, results_dir):
    dat = np.load(seg_file, allow_pickle=True).item()
    masks = dat['masks']
    outlines = dat['outlines']
    colors = np.array(dat['colors'])

    # Assuming you have the original image saved somewhere or loaded it
    # img = io.imread('img.tif')

    # Convert masks to label PNG
    label_image = np.zeros(masks.shape[:2], dtype=np.uint8)
    for idx, mask in enumerate(masks):
        label_image[mask > 0] = idx + 1  # Increment label by 1 to avoid 0 label

    # Save the label PNG
    label_filename = os.path.splitext(os.path.basename(seg_file))[0] + '_label.png'
    label_path = os.path.join(results_dir, label_filename)
    plt.imsave(label_path, label_image, cmap='gray')

    # Optionally, plot masks with outlines overlaid
    mask_RGB = plot.mask_overlay(img, masks, colors=colors)
    plt.imshow(mask_RGB)
    for o in outlines:
         plt.plot(o[:, 0], o[:, 1], color='r')
    plt.savefig(label_path.replace('_label.png', '_overlay.png'))  # Save overlay image

    return label_path

def convert_all_segs_to_labels(seg_directory, results_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    seg_files = [f for f in os.listdir(seg_directory) if f.endswith('_seg.npy')]
    for seg_file in seg_files:
        label_path = convert_seg_to_label(os.path.join(seg_directory, seg_file), results_dir)
        print(f"Converted {seg_file} to {label_path}")

# Example usage
seg_directory = r'Z:\Urte\PhD\3D modeller\training data\Temporal model'
results_directory = r'Z:\Urte\PhD\3D modeller\training data\Temporal model\results'
convert_all_segs_to_labels(seg_directory, results_directory)
