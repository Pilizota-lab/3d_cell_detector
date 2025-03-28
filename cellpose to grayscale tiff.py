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
    if os.path.isdir(path_to_model):
        raise ValueError(f"The path '{path_to_model}' is a directory. Please provide the path to a specific model file.")
    print(f"Loading model from {path_to_model}")
    model = models.CellposeModel(pretrained_model=path_to_model, gpu=True)
    return model

def load_images(image_files):
    print(f"Loading images")
    images = []
    original_dims = []
    image_names = []

    for image_file in tqdm(image_files, desc="Loading Images"):
        try:
            print(f"Reading image: {image_file}")
            image = io.imread(image_file)
            original_dims.append(image.shape)
            new_height, new_width = image.shape[0] // 3, image.shape[1] // 3
            image_resized = transform.resize(image, (new_height, new_width), anti_aliasing=True)
            images.append(image_resized)
            image_names.append(os.path.splitext(os.path.basename(image_file))[0])  # Store the image name without extension
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    print(f"Loaded {len(images)} images.")
    return images, original_dims, image_names

def save_masks_and_npy(masks, image_names, save_directory):
    npy_dir = os.path.join(save_directory, 'npy')
    Path(npy_dir).mkdir(parents=True, exist_ok=True)
    
    for mask, image_name in tqdm(zip(masks, image_names), total=len(masks), desc="Saving NPY files"):
        try:
            mask = np.squeeze(mask)
            if mask.ndim > 2:
                mask = mask[..., 0]

            npy_path = os.path.join(npy_dir, f'{image_name}_seg.npy')
            np.save(npy_path, mask)
        except Exception as e:
            print(f"Error saving NPY file for {image_name}: {e}")
    print(f"Saved NPY files for {len(masks)} masks.")

def save_masks_overlay(images, masks, original_dims, image_names, save_directory):
    print(f"Saving masks and overlays to {save_directory}")
    masks_dir = os.path.join(save_directory, 'masks')
    overlay_dir = os.path.join(save_directory, 'overlay')
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(overlay_dir).mkdir(parents=True, exist_ok=True)
    
    for image, mask, orig_dim, image_name in tqdm(zip(images, masks, original_dims, image_names), total=len(images), desc="Saving Masks and Overlays"):
        try:
            # Ensure mask is 2D
            mask = np.squeeze(mask)
            if mask.ndim > 2:
                mask = mask[..., 0]

            mask_resized_back = transform.resize(mask, (orig_dim[0], orig_dim[1]), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)
            
            mask_path = os.path.join(masks_dir, f'{image_name}_label.tiff')
            overlay_path = os.path.join(overlay_dir, f'{image_name}_overlay.tiff')
            
            plt.imsave(mask_path, mask_resized_back, cmap='gray')
            
            image_resized_back = transform.resize(image, (orig_dim[0], orig_dim[1]), anti_aliasing=True)
            
            labeled_image = color.label2rgb(mask_resized_back, image_resized_back, alpha=0.2)
            plt.imsave(overlay_path, labeled_image)
        except Exception as e:
            print(f"Error saving mask or overlay for {image_name}: {e}")
    print(f"Saved masks and overlays for {len(images)} images.")

def process_files(image_files, model, model_identifier, base_directory):
    print(f"Processing files in {base_directory}")
    try:
        images, original_dims, image_names = load_images(image_files)
        if not images:
            print(f"No images to process in directory: {base_directory}")
            return 0
        results_dir = Path(base_directory) / f'results_model_{model_identifier}'
        results_dir.mkdir(parents=True, exist_ok=True)
        
        masks, flows, styles = model.eval(images, diameter=30, channels=[[0, 0]], compute_masks=True)
        save_masks_overlay(images, masks, original_dims, image_names, results_dir)
        save_masks_and_npy(masks, image_names, results_dir)

        count_filename = results_dir / 'cell_count.txt'
        number_of_cells = sum(np.max(mask) for mask in masks)
        with open(count_filename, 'w') as file:
            file.write(str(number_of_cells))
        print(f"Detected {number_of_cells} cells in directory {base_directory}.")
        return number_of_cells
    except Exception as e:
        print(f"Error processing files in directory {base_directory}: {e}")
        return 0

def main(base_directory, model_path, model_identifier='temporal_4'):
    try:
        print(f"Base directory: {base_directory}")
        print(f"Model path: {model_path}")
        print(f"Model identifier: {model_identifier}")

        model = load_working_model(model_path)
        print(f"Model loaded successfully.")

        if not os.path.isdir(base_directory):
            print(f"The base directory '{base_directory}' does not exist or is not a directory.")
            return

        image_files = [os.path.join(base_directory, f) for f in os.listdir(base_directory) if f.endswith('.tiff')]
        if image_files:
            print(f"Found {len(image_files)} TIFF files in the base directory.")
            process_files(image_files, model, model_identifier, base_directory)
        else:
            print(f"No TIFF files found in the base directory.")
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    base_directory = '/home/urte/MEDIAR-main/test_images/random_images'
    model_path = '/home/urte/3D modeller/3d_cell_detector/models/temporal_4'  # Update this line
    main(base_directory, model_path)
