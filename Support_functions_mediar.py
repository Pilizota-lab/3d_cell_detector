import os
import re
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte, measure
from skimage.color import label2rgb
import hdbscan
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm, trange
from scipy.ndimage import label
from scipy.spatial.distance import cdist


def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def update_processed_directories_log(directory, log_file_path):
    """Update the log file with the processed directory."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(directory + '\n')
    print(f"Directory {directory} added to log.")


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

        # Ensure the mask is integer type for label2rgb
        mask = mask.astype(int)

        # Create an overlay image using label2rgb with a specified alpha
        overlay_image = label2rgb(mask, image=image, bg_label=0, alpha=0.2)

        # Save the overlay image
        overlay_path = os.path.join(output_folder, f'overlay_{image_file.replace(".tiff", ".png")}')
        try:
            io.imsave(overlay_path, img_as_ubyte(overlay_image))
            print(f"Overlay saved to {overlay_path}")
        except Exception as e:
            print(f"Failed to save overlay for {image_file}: {e}")
            
def process_directory(directory, config_path):
    """
    Processes a directory of images using outputs from MEDIAR model.
    Assumes MEDIAR saves output in a predictable location as specified in the config.
    """
    results_dir = os.path.join(directory, 'results')
    if not os.path.exists(results_dir):
        print(f"No results directory found for {directory}, skipping overlay and projection generation.")
        return None, 0  # Return zero cells as the process did not complete

    mask_files = [f for f in os.listdir(results_dir) if f.endswith('_label.tiff')]
    masks = [io.imread(os.path.join(results_dir, f)) for f in mask_files]

    # Create 3D Projection and Plot
    create_3d_stack(masks, results_dir)

    # Perform clustering analysis on masks and create the final 3D plot
    number_of_cells = run_cluster_analysis_on_masks(masks, results_dir)

    # Create overlays after successful mask creation
    create_overlays(mask_files, directory, results_dir)

def stitch3D_simple(masks, stitch_threshold=0.25):
    """
    Stitch 2D masks into a 3D volume using a stitch_threshold on IOU.
    """
    mmax = masks[0].max()
    empty = 0

    for i in range(len(masks) - 1):
        iou = compute_iou(masks[i + 1], masks[i])
        if not iou.size and empty == 0:
            masks[i + 1] = masks[i + 1]
            mmax = masks[i + 1].max()
        elif not iou.size and not empty == 0:
            icount = masks[i + 1].max()
            istitch = np.arange(mmax + 1, mmax + icount + 1, 1, int)
            mmax += icount
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = np.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = np.arange(mmax + 1, mmax + len(ino) + 1, 1, int)
            mmax += len(ino)
            istitch = np.append(np.array(0), istitch)
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1

    return masks

def compute_iou(mask1, mask2):
    """Compute the Intersection over Union (IoU) between two masks."""
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def run_cluster_analysis_on_masks(masks, save_directory):
    """Performs clustering on detected cells from mask files using HDBSCAN."""
    centers = []
    for i, mask in enumerate(masks):
        labeled_mask = measure.label(mask > 0, connectivity=mask.ndim)
        for region in measure.regionprops(labeled_mask):
            center_y, center_x = region.centroid
            centers.append([center_x, center_y, i])

    data_array = np.array(centers)
    
    # Perform initial clustering
    cluster = hdbscan.HDBSCAN(min_cluster_size=5).fit(data_array)
    labels = cluster.labels_

    # Correct clustering by ensuring no duplicates in x and y within the same cluster
    unique_labels = set(labels) - {-1}  # Exclude noise
    corrected_labels = labels.copy()

    for k in unique_labels:
        indices = np.where(labels == k)[0]
        cluster_points = data_array[indices]
        
        # Calculate pairwise distances between points in the cluster
        distances = cdist(cluster_points[:, :2], cluster_points[:, :2])
        np.fill_diagonal(distances, np.inf)  # Ignore self-distances by setting them to infinity
        
        # Ensure no two points in the same (x, y) position in the same cluster
        for i, idx in enumerate(indices):
            duplicate_indices = np.where((distances[i] == 0) & (cluster_points[:, 2] != cluster_points[i, 2]))[0]
            for dup in duplicate_indices:
                corrected_labels[indices[dup]] = max(corrected_labels) + 1

    unique_corrected_labels = set(corrected_labels) - {-1}  # Recalculate unique labels

    # Plot the corrected clusters
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in unique_corrected_labels:
        class_member_mask = (corrected_labels == k)
        xyz = data_array[class_member_mask]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=f'Cell {k}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f'Clustered Cells: {len(unique_corrected_labels)} Cells')
    plt.legend()
    plt.savefig(os.path.join(save_directory, 'clustered_cells_3d_plot.png'))
    plt.show()

    return len(unique_corrected_labels)

def save_results_to_excel(folder_names, cell_counts, output_path):
    """Saves results to an Excel file."""
    results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

