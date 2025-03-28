
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

def plot_masks_from_directory(directory):
    # List all image files in the directory
    # Includes common image file extensions for broader support
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp')
    mask_files = [f for f in os.listdir(directory) if f.endswith(supported_extensions)]
    mask_files.sort()  # Optional: sort the files alphabetically

    # Plot each mask
    for mask_file in tqdm(mask_files, desc="Plotting Masks"):
        file_path = os.path.join(directory, mask_file)
        mask = imageio.imread(file_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(mask, cmap='nipy_spectral')  # Use a colormap that best fits the image type
        plt.colorbar()
        plt.title(mask_file)
        plt.show()

# Example usage: Specify the path to your directory containing mask image files
#directory_path = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\CellSeg\Labelled\labels'
directory_path= '/home/urte/MEDIAR-main/trained images/just missing labels'
plot_masks_from_directory(directory_path)
