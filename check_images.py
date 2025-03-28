import os
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte
import numpy as np

def overlay_masks(image_folder, mask_folder, output_folder):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # List all the images in the image folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.tiff')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        mask_path = os.path.join(mask_folder, image_file.replace('.tiff', '_label.tiff'))

        # Load the original image and the mask
        image = imread(image_path)
        if os.path.exists(mask_path):
            mask = imread(mask_path)
        else:
            print(f"No mask found for {image_file}, skipping.")
            continue

        # Ensure the mask is boolean
        mask = mask.astype(bool)

        # Create an overlay image
        overlay_image = image.copy()
        overlay_image[mask] = [255, 0, 0]  # Overlay color is red

        # Save the overlay image
        overlay_path = os.path.join(output_folder, f'overlay_{image_file}')
        imsave(overlay_path, img_as_ubyte(overlay_image))
        print(f"Overlay saved to {overlay_path}")

# Define the paths
image_folder = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images'
mask_folder = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\results'
output_folder = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\overlays'

# Call the function
overlay_masks(image_folder, mask_folder, output_folder)
