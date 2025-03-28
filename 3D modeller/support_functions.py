from cellpose import models
from cellpose.io import imread
from skimage import io, measure, color, segmentation, transform
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
import pandas as pd
import tensorflow as tf

def check_gpu_availability():
    physical_devices = tf.config.list_physical_devices('GPU')
    return len(physical_devices) > 0

print(check_gpu_availability())

def load_working_model(path_to_model):
    # Load and Run Cellpose Model
    model =  models.CellposeModel(pretrained_model=path_to_model, gpu=False)
    return model

#Natural sorting function
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split('([0-9]+)', s)]


def load_images(path_to_images):
    img_dir = path_to_images
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]
    sorted_image_files= sorted (image_files, key=natural_sort_key)
    images = []
    for image_file in sorted_image_files:
        image_path = os.path.join(img_dir, image_file)
        image = io.imread(image_path)
        # Calculate the new dimensions
        new_height = image.shape[0] // 3
        new_width = image.shape[1] // 3

        # Resize the image
        shrunken_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
        images.append(shrunken_image)
    return images

def save_masks_overlay(images, masks, save_directory ):
    
    # Create image mask overlay and save images
    mask_dir = (f'{save_directory}/masks').mkdir(parents=True, exist_ok=True)
    lab_img_dir = (f'{save_directory}/overlay').mkdir(parents=True, exist_ok=True)
   
    zip_masks = zip(images, masks)
    labelled_images = [color.label2rgb(mask, img, alpha = 0.2) for img, mask in zip_masks]
    
    zip_masks = zip(images, masks)
    labelled_images = [color.label2rgb(mask, img, alpha = 0.2) for img, mask in zip_masks]

   

    for i, image in enumerate(masks):
        image_filename = f'image_{i}.png'  # You can adjust the filename pattern
        image_path = os.path.join(save_directory, mask_dir, image_filename)
        plt.imsave(image_path, image)

    l

    for i, image in enumerate(labelled_images):
        image_filename = f'image_{i}.png'  # You can adjust the filename pattern
        image_path = os.path.join(save_directory, lab_img_dir, image_filename)
        plt.imsave(image_path, image)
    
    return labelled_images


#Create 3d stack

def create_3d_stack(masks):
    Path(f"{save_directory}/3d_images").mkdir(parents=True, exist_ok=True)
    # List of mask images
    mask_images = masks  # List of your mask images

    # Create a 3D projection

    # Create a 3D-like scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Loop through the masks and create scatter points for object locations
    for i, mask in enumerate(mask_images):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            z_coords = np.full((object_indices.shape[0],), i)
            ax.scatter(object_indices[:, 1], object_indices[:, 0], z_coords, c=np.random.rand(3,), marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Slice (Object Index)')

    plt.savefig(f"{save_directory}/3d_images/3d_image.png", dpi=300)


def create_3d_projection(masks):
    Path(f"{save_directory}/3d_images").mkdir(parents=True, exist_ok=True)

    # List of mask images
    mask_images = masks

    # Initialize a dictionary to store unique colors for labels
    label_to_color = {}

    # Loop through the masks and create scatter points for object locations
    for i in range(1, len(mask_images)):
        prev_mask = mask_images[i - 1]
        curr_mask = mask_images[i]
        
        # Label objects in the previous and current masks
        labeled_prev_mask, num_prev_labels = measure.label(prev_mask, return_num=True)
        labeled_curr_mask, num_curr_labels = measure.label(curr_mask, return_num=True)
        
        # Loop through objects in the current mask
        for label_curr in range(1, num_curr_labels + 1):
            object_curr = labeled_curr_mask == label_curr
            
            # Calculate overlap with objects in the previous mask
            overlaps = labeled_prev_mask * object_curr
            overlapping_labels = np.unique(overlaps)
            overlapping_labels = overlapping_labels[overlapping_labels != 0]
            
            # Loop through overlapping objects and check overlap ratio
            for label_prev in overlapping_labels:
                overlap_pixels = np.sum(overlaps == label_prev)
                total_pixels = np.sum(labeled_prev_mask == label_prev)
                overlap_ratio = overlap_pixels / total_pixels
                
                if overlap_ratio >= 0.6:
                    if label_prev not in label_to_color:
                        label_to_color[label_prev] = np.random.rand(3,)
                    
                    color = label_to_color[label_prev]
                    object_indices = np.argwhere(labeled_curr_mask == label_curr)
                    
                    plt.scatter(object_indices[:, 1], object_indices[:, 0], c=[color], marker='o')

        plt.xlabel('X')
        plt.ylabel('Y')

        plt.savefig(f"{save_directory}/3d_images/3d_projection.png", dpi = 300)


def count_objects(masks, save_directory):
    # Initialize arrays to store center points
    centers = []

    # Load masks and determine centers
    for i, mask_file in enumerate(masks):
        mask = masks[i]
        # Label connected components in the mask
        labeled_mask = measure.label(mask > 0)
        
        # Calculate center of mass for each labeled region
        for region in measure.regionprops(labeled_mask):
            center_y, center_x = region.centroid
            centers.append([center_x, center_y, i])

    # Convert centers list to array
    centers = np.array(centers)

    # Define grouping distance (25 pixels by 25 pixels); I am justifying this as I have cells that move each slice
    grouping_distance = 20
    # Initialize dictionary to store color assignments
    color_assignments = {}
    current_color = 0

    # Assign colors to center points based on grouping
    for center in centers:
        found_group = False
        for color, points in color_assignments.items():
            if any(np.linalg.norm(center[:2] - point[:2]) <= grouping_distance for point in points):
                color_assignments[color].append(center)
                found_group = True
                break
        if not found_group:
            color_assignments[current_color] = [center]
            current_color += 1

    # Create an XYZ scatter plot with unique colors for each group
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for color, points in color_assignments.items():
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=f'Group {color}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    total_entries = len(ax.legend().legendHandles)
    ax.legend([f'Number of unique objects: {total_entries}'])                 

    plt.savefig(f"{save_directory}/3d_images/object_count.png", dpi = 300)
    return total_entries

def save_to_excel(folders, cell_counts, output_filename="cell_counts_15_07.xlsx"):
    df = pd.DataFrame({
        "Folder": folders,
        "Cell Count": cell_counts
    })
    df.to_excel(output_filename, index=False, engine='openpyxl')


# def plot_growth_curve(folders, cell_counts):