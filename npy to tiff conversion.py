import os
import re
import numpy as np
from pathlib import Path
from skimage import io, transform, color
import matplotlib.pyplot as plt
from cellpose import models
from tqdm import tqdm

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_working_model(path_to_model):
    print(f"Loading model from {path_to_model}")
    model = models.CellposeModel(pretrained_model=path_to_model, gpu=True)
    return model

def load_images(path_to_images):
    print(f"Loading images from {path_to_images}")
    image_files = sorted([f for f in os.listdir(path_to_images) if f.endswith('.tiff')], key=numerical_sort_key)
    images = []
    original_dims = []
    for image_file in tqdm(image_files, desc="Loading Images"):
        image_path = os.path.join(path_to_images, image_file)
        image = io.imread(image_path)
        original_dims.append(image.shape)
        new_height, new_width = image.shape[0] // 3, image.shape[1] // 3
        image_resized = transform.resize(image, (new_height, new_width), anti_aliasing=True)
        images.append(image_resized)
    print(f"Loaded {len(images)} images.")
    return images, original_dims

def save_masks_overlay(images, masks, original_dims, save_directory):
    print(f"Saving masks and overlays to {save_directory}")
    masks_dir = os.path.join(save_directory, 'masks')
    overlay_dir = os.path.join(save_directory, 'overlay')
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(overlay_dir).mkdir(parents=True, exist_ok=True)
    
    for i, (image, mask, orig_dim) in enumerate(zip(images, masks, original_dims)):
        # Ensure mask is 2D
        mask = np.squeeze(mask)
        if mask.ndim > 2:
            mask = mask[..., 0]
        
        # Debug print to check dimensions
        print(f"Original mask shape: {mask.shape}")
        print(f"Original image shape: {orig_dim}")

        mask_resized_back = transform.resize(mask, (orig_dim[0], orig_dim[1]), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
        
        # Debug print to check dimensions after resizing
        print(f"Resized mask shape: {mask_resized_back.shape}")
        
        mask_path = os.path.join(masks_dir, f'image_{i}.tiff')
        overlay_path = os.path.join(overlay_dir, f'image_{i}.tiff')
        
        plt.imsave(mask_path, mask_resized_back, cmap='gray')
        
        image_resized_back = transform.resize(image, (orig_dim[0], orig_dim[1]), anti_aliasing=True)
        
        labeled_image = color.label2rgb(mask_resized_back, image_resized_back, alpha=0.2)
        plt.imsave(overlay_path, labeled_image)
    print(f"Saved masks and overlays for {len(images)} images.")

def process_directory(directory, model, model_identifier):
    print(f"Processing directory {directory}")
    images, original_dims = load_images(directory)
    results_dir = Path(directory) / f'results_model_{model_identifier}'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    masks, flows, styles = model.eval(images, diameter=30, channels=[[0, 0]], compute_masks=True)
    save_masks_overlay(images, masks, original_dims, results_dir)

    count_filename = results_dir / 'cell_count.txt'
    number_of_cells = sum(np.max(mask) for mask in masks)
    with open(count_filename, 'w') as file:
        file.write(str(number_of_cells))
    print(f"Detected {number_of_cells} cells in directory {directory}.")
    return number_of_cells

def main(base_directory, model_path, model_identifier='temporal_4'):
    model = load_working_model(model_path)
    for directory_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, directory_name)
        if os.path.isdir(dir_path):
            print(f"Processing directory: {dir_path}")
            number_of_cells = process_directory(dir_path, model, model_identifier)
            print(f"Processed directory '{dir_path}' with {number_of_cells} cells detected.")

if __name__ == "__main__":
    base_directory = r'Z:\Urte\PhD\3D modeller\training data\Temporal model'
    output_directory = r'Z:\Urte\PhD\3D modeller\training data\Temporal model\CellSeg\labels'
    main(base_directory, output_directory)
