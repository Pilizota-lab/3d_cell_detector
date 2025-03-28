import os
import numpy as np
from skimage import io, measure, img_as_ubyte, img_as_float ,exposure
from skimage.color import label2rgb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def overlay_masks(image_folder, mask_folder, output_folder):
    """Generates overlay images and saves them as PNG files."""
    os.makedirs(output_folder, exist_ok=True)
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.tiff')]
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file.replace('.tiff', '_label.tiff'))

        # Ensure both image and mask exist
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"Image or mask not found for {image_file}, skipping.")
            continue

        # Load the original image and the mask
        try:
            image = io.imread(image_path)
            mask = io.imread(mask_path)
        except Exception as e:
            print(f"Error loading image or mask for {image_file}: {e}")
            continue

        # Check if image and mask dimensions match
        if image.shape[:2] != mask.shape[:2]:
            print(f"Dimension mismatch for {image_file}, skipping.")
            continue

        # Ensure the mask is integer type for label2rgb
        mask = mask.astype(int)

        # Adjust intensity values for better visualization
        image_rescaled = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))

        # Convert the image to float type
        image_float = img_as_float(image_rescaled)

        # Create an overlay image using label2rgb with a specified alpha
        overlay_image = label2rgb(mask, image=image_float, bg_label=0, alpha=0.2)

        # Convert the overlay image to ubyte for saving
        overlay_image_ubyte = img_as_ubyte(overlay_image)

        # Save the overlay image as PNG
        overlay_path = os.path.join(output_folder, f'overlay_{image_file.replace(".tiff", ".png")}')
        try:
            io.imsave(overlay_path, overlay_image_ubyte)
            print(f"Overlay saved to {overlay_path}")
        except Exception as e:
            print(f"Failed to save overlay for {image_file}: {e}")

def create_3d_stack(masks, save_directory):
    stack_dir = os.path.join(save_directory, "3d_images")
    Path(stack_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            z_coords = np.full((object_indices.shape[0],), i)
            ax.scatter(object_indices[:, 1], object_indices[:, 0], z_coords, c=np.random.rand(3,), marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Slice (Object Index)')

    plt.savefig(os.path.join(stack_dir, "3d_image.png"), dpi=300)
    plt.show()

def process_directories(base_directory):
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir = os.path.join(dir_path, 'results')
        if os.path.isdir(dir_path) and os.path.exists(results_dir):
            mask_files = [f for f in os.listdir(results_dir) if f.endswith('_label.tiff')]
            masks = [io.imread(os.path.join(results_dir, f)) for f in mask_files]

            overlay_output_folder = os.path.join(results_dir, 'overlay')
            # Generate overlays
            overlay_masks(dir_path, results_dir, overlay_output_folder)

            # Generate 3D plot
            create_3d_stack(masks, results_dir)

if __name__ == "__main__":
    base_directory = "/home/urte/MEDIAR-main/growth curves test images"
    process_directories(base_directory)
