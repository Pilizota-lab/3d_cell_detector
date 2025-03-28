import os
from skimage.io import imread, imsave
from skimage import img_as_ubyte, color
import numpy as np

def overlay_masks(image_folder, mask_folder, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all the images in the image folder
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
            image = imread(image_path)
            mask = imread(mask_path)
        except Exception as e:
            print(f"Error loading image or mask for {image_file}: {e}")
            continue

        # Ensure the mask is integer type for label2rgb
        mask = mask.astype(int)

        # Create an overlay image using label2rgb with a specified alpha
        overlay_image = color.label2rgb(mask, image=image, bg_label=0, alpha=0.2)

        # Save the overlay image
        overlay_path = os.path.join(output_folder, f'overlay_{image_file}')
        try:
            imsave(overlay_path, img_as_ubyte(overlay_image))
            print(f"Overlay saved to {overlay_path}")
        except Exception as e:
            print(f"Failed to save overlay for {image_file}: {e}")

# Define the paths
image_folder = "/home/urte/MEDIAR-main/test_images/24_03_03_23_21_31"
mask_folder = "/home/urte/MEDIAR-main/test_images/24_03_03_23_21_31/results"
output_folder = "/home/urte/MEDIAR-main/test_images/24_03_03_23_21_31/results/overlays"

# Call the function
overlay_masks(image_folder, mask_folder, output_folder)
