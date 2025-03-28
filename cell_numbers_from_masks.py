import os
import numpy as np
from skimage import io, measure, color
import hdbscan
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

# Use the Agg backend for Matplotlib to avoid Tkinter issues
matplotlib.use('Agg')

# Parameters
DISTANCE_THRESHOLD = 5
MIN_CLUSTER_SIZE = 6
MAX_GAP = 20

def assign_cell_numbers(masks, distance_threshold, max_gap):
    max_label = 0
    labeled_masks = []
    cell_centers = []

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True, connectivity=1)
        new_labels = np.zeros_like(labeled_mask)

        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            center = np.mean(object_indices, axis=0)
            if i == 0:
                max_label += 1
                new_labels[labeled_mask == label] = max_label
                cell_centers.append((center, max_label, i))
            else:
                prev_centers = [c for c in cell_centers if i - c[2] <= max_gap]
                if prev_centers:
                    distances = cdist([center], [c[0] for c in prev_centers])
                    min_dist = distances.min()
                    if min_dist < distance_threshold:
                        min_index = distances.argmin()
                        new_labels[labeled_mask == label] = prev_centers[min_index][1]
                    else:
                        max_label += 1
                        new_labels[labeled_mask == label] = max_label
                        cell_centers.append((center, max_label, i))
                else:
                    max_label += 1
                    new_labels[labeled_mask == label] = max_label
                    cell_centers.append((center, max_label, i))

        labeled_masks.append(new_labels)

    return labeled_masks, max_label

def run_cluster_analysis_on_masks(masks, save_directory, distance_threshold, min_cluster_size, max_gap):
    labeled_masks, num_cells = assign_cell_numbers(masks, distance_threshold, max_gap)

    centers = []
    for i, labeled_mask in enumerate(labeled_masks):
        for region in measure.regionprops(labeled_mask):
            center_y, center_x = region.centroid
            centers.append([center_x, center_y, i])

    if not centers:
        print(f"No centers found in masks for {save_directory}.")
        return 0, [], []

    data_array = np.array(centers)
    if data_array.ndim != 2 or data_array.shape[1] != 3:
        print(f"Data array has unexpected shape: {data_array.shape}")
        return 0, [], []

    # Initial clustering
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data_array)
    labels = cluster.labels_

    # Correct clustering
    corrected_labels = labels.copy()
    unique_labels = set(labels) - {-1}
    max_label = max(labels)

    for k in unique_labels:
        indices = np.where(labels == k)[0]
        cluster_points = data_array[indices]
        xy_points = cluster_points[:, :2]
        z_points = cluster_points[:, 2]

        for i in range(len(xy_points)):
            for j in range(i + 1, len(xy_points)):
                if np.linalg.norm(xy_points[i] - xy_points[j]) < distance_threshold and z_points[i] != z_points[j]:
                    max_label += 1
                    corrected_labels[indices[j]] = max_label

    # Save clustering results
    cluster_save_path = os.path.join(save_directory, '3d_clusters')
    os.makedirs(cluster_save_path, exist_ok=True)
    np.save(os.path.join(cluster_save_path, 'corrected_labels.npy'), corrected_labels)
    np.save(os.path.join(cluster_save_path, 'data_array.npy'), data_array)

    return len(set(corrected_labels) - {-1}), corrected_labels, labeled_masks

def count_cells_per_slice(masks):
    cell_counts = []
    labeled_masks = []

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True, connectivity=1)
        labeled_masks.append(labeled_mask)
        cell_counts.append(num_labels)

    return labeled_masks, cell_counts

def visualize_cell_numbers(masks, labeled_masks, cell_counts, results_dir, dir_name):
    cell_count_save_path = os.path.join(results_dir, 'cell_counts')
    os.makedirs(cell_count_save_path, exist_ok=True)

    for i, (mask, labeled_mask, count) in enumerate(zip(masks, labeled_masks, cell_counts)):
        overlay = color.label2rgb(labeled_mask, image=mask, bg_label=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f'{dir_name} - Slice {i+1} - Cells: {count}')
        plt.axis('off')
        save_path = os.path.join(cell_count_save_path, f'overlay_slice_{i+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overlay to {save_path}")

def process_directories(base_directory, distance_threshold, min_cluster_size, max_gap):
    folder_names = []
    cell_counts = []

    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir = os.path.join(dir_path, 'results')
        if os.path.isdir(dir_path) and os.path.exists(results_dir):
            mask_files = [f for f in os.listdir(results_dir) if f.endswith('_label.tiff')]
            masks = [io.imread(os.path.join(results_dir, f)) for f in mask_files]

            number_of_cells, corrected_labels, labeled_masks = run_cluster_analysis_on_masks(masks, results_dir, distance_threshold, min_cluster_size, max_gap)
            folder_names.append(dir_name)
            cell_counts.append(number_of_cells)
            print(f"Processed directory '{dir_name}' with {number_of_cells} cells detected.")

            # Count cells per slice and visualize
            _, cell_counts_per_slice = count_cells_per_slice(masks)
            visualize_cell_numbers(masks, labeled_masks, cell_counts_per_slice, results_dir, dir_name)

    save_results_to_excel(folder_names, cell_counts, os.path.join(base_directory, 'cell_counts_summary.xlsx'))

def save_results_to_excel(folder_names, cell_counts, output_path):
    results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    base_directory = "/home/urte/MEDIAR-main/growth curves test images"
    process_directories(base_directory, DISTANCE_THRESHOLD, MIN_CLUSTER_SIZE, MAX_GAP)
