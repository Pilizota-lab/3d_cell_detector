import os
import numpy as np
import imageio
from tqdm import tqdm

def analyze_masks(directory):
    # List all image files in the directory
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.npy')
    mask_files = [f for f in os.listdir(directory) if f.endswith(supported_extensions)]
    mask_files.sort()  # Optional: sort files alphabetically

    # Analyze each mask file
    analysis_results = []
    for mask_file in tqdm(mask_files, desc="Analyzing Masks"):
        file_path = os.path.join(directory, mask_file)
        
        # Load the mask file depending on its extension
        if mask_file.endswith('.npy'):
            mask = np.load(file_path)
        else:
            mask = imageio.imread(file_path)
        
        unique_values = np.unique(mask)
        image_type = mask.dtype
        
        analysis_results.append({
            "file_name": mask_file,
            "path": file_path,
            "unique_values": unique_values,
            "data_type": image_type,
            "shape": mask.shape
        })
        
        # Optionally, you can print each result to check them one by one
        print(f"File: {mask_file}")
        print(f"Path: {file_path}")
        print(f"Unique Values: {unique_values}")
        print(f"Data Type: {image_type}")
        print(f"Shape: {mask.shape}")
        print("-" * 40)
    
    return analysis_results

# Example usage
directory_path = r'/home/urte/MEDIAR-main/test_images/random_images/results/masks'
results = analyze_masks(directory_path)
