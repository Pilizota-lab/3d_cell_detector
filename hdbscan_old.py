import os
import numpy as np
from skimage import io, measure, color
import hdbscan
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go

# Use the Agg backend for Matplotlib to avoid Tkinter issues
matplotlib.use('Agg')

# Parameters
DISTANCE_THRESHOLD = 10  # distance at which two masks are considered the same cell (in pixels)
MIN_CLUSTER_SIZE = 6  # min cluster size for a cluster to be considered valid
MAX_GAP = 25  # how many times a cell is to reoccur throughout the stack
SKIP_ALLOWED = 1  # cell to be counted as the same cell even if one segmentation mask is missing

def assign_cell_numbers(masks, distance_threshold, max_gap, skip_allowed):
    max_label = 0  # tracks highest cell number
    labeled_masks = []
    cell_centers = []

    for i, mask in enumerate(masks):  # loop through each mask
        labeled_mask, num_labels = measure.label(mask, return_num=True, connectivity=1)  # labels components of the current mask
        new_labels = np.zeros_like(labeled_mask)  # new label array for the current mask

        for label in range(1, num_labels + 1):  # loop through each labeled object
            object_indices = np.argwhere(labeled_mask == label)  # finds indices of the current label
            center = np.mean(object_indices, axis=0)  # calculates centroid of the object
            if i == 0:  # for the first slice; new label to record the center
                max_label += 1
                new_labels[labeled_mask == label] = max_label
                cell_centers.append((center, max_label, i))
            else:
                prev_centers = [c for c in cell_centers if i - c[2] <= max_gap + skip_allowed]  # for each slice after the first, find previous centers within the allowed gap and skip range
                if prev_centers:
                    distances = cdist([center], [c[0] for c in prev_centers])  # calculate distances to previous centers
                    min_dist = distances.min()  # find the minimum distance
                    if min_dist < distance_threshold:  # check if within the distance threshold
                        min_index = distances.argmin()
                        new_labels[labeled_mask == label] = prev_centers[min_index][1]  # assign existing label
                    else:
                        max_label += 1
                        new_labels[labeled_mask == label] = max_label
                        cell_centers.append((center, max_label, i))  # assign new label
                else:
                    max_label += 1
                    new_labels[labeled_mask == label] = max_label
                    cell_centers.append((center, max_label, i))  # assign new label if no previous centers

        labeled_masks.append(new_labels)

    return labeled_masks, cell_centers

def create_interactive_overlay(masks, labeled_masks, cell_centers, results_dir, dir_name):
    cell_count_save_path = os.path.join(results_dir, 'cell_counts')
    os.makedirs(cell_count_save_path, exist_ok=True)

    for i, (mask, labeled_mask) in enumerate(zip(masks, labeled_masks)):
        overlay = color.label2rgb(labeled_mask, image=mask, bg_label=0)
        
        fig = go.Figure()
        scatter_data = []

        for center, label, slice_idx in cell_centers:
            if slice_idx == i:
                centroid = center
                pixel_values = mask[labeled_mask == label]
                mean_pixel_value = np.mean(pixel_values)

                scatter_data.append(go.Scatter(
                    x=[centroid[1]],
                    y=[centroid[0]],
                    mode='markers+text',
                    marker=dict(size=10, color='blue'),
                    text=[f'Label: {label}<br>Coordinates: ({centroid[1]:.1f}, {centroid[0]:.1f})<br>Mean Pixel Value: {mean_pixel_value:.1f}'],
                    hoverinfo='text'
                ))

        fig.add_trace(go.Image(z=overlay))
        for scatter in scatter_data:
            fig.add_trace(scatter)

        fig.update_layout(
            title=f'{dir_name} - Slice {i+1} - Cells: {len(scatter_data)}',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            hovermode='closest'
        )

        save_path = os.path.join(cell_count_save_path, f'interactive_overlay_slice_{i+1}.png')
        fig.write_html(save_path)
        print(f"Saved interactive overlay to {save_path}")

def run_cluster_analysis_on_masks(masks, save_directory, distance_threshold, min_cluster_size, max_gap, skip_allowed):
    labeled_masks, cell_centers = assign_cell_numbers(masks, distance_threshold, max_gap, skip_allowed)

    centers = []
    for center, label, slice_idx in cell_centers:
        centers.append([center[1], center[0], slice_idx])  # Swapping x and y for better visualization

    if not centers:
        print(f"No centers found in masks for {save_directory}.")
        return 0, [], [], []

    data_array = np.array(centers)
    if data_array.ndim != 2 or data_array.shape[1] != 3:
        print(f"Data array has unexpected shape: {data_array.shape}")
        return 0, [], [], []

    # Perform initial clustering
    cluster = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(data_array)
    labels = cluster.labels_

    return len(set(labels) - {-1}), labels, labeled_masks, cell_centers

def count_cells_per_slice(masks):
    cell_counts = []
    labeled_masks = []

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True, connectivity=1)
        labeled_masks.append(labeled_mask)
        cell_counts.append(num_labels)

    return labeled_masks, cell_counts

def process_directories(base_directory, distance_threshold, min_cluster_size, max_gap, skip_allowed):
    folder_names = []
    cell_counts = []

    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir = os.path.join(dir_path, 'results')
        if os.path.isdir(dir_path) and os.path.exists(results_dir):
            mask_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_label.tiff')],
                                key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))
            masks = [io.imread(os.path.join(results_dir, f)) for f in mask_files]

            number_of_cells, labels, labeled_masks, cell_centers = run_cluster_analysis_on_masks(masks, results_dir, distance_threshold, min_cluster_size, max_gap, skip_allowed)
            folder_names.append(dir_name)
            cell_counts.append(number_of_cells)
            print(f"Processed directory '{dir_name}' with {number_of_cells} cells detected.")

            # Create interactive overlays for each slice
            create_interactive_overlay(masks, labeled_masks, cell_centers, results_dir, dir_name)

    save_results_to_excel(folder_names, cell_counts, os.path.join(base_directory, 'cell_counts_summary.xlsx'))

def save_results_to_excel(folder_names, cell_counts, output_path):
    results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    base_directory = "/home/urte/MEDIAR-main/growth curves test images"
    process_directories(base_directory, DISTANCE_THRESHOLD, MIN_CLUSTER_SIZE, MAX_GAP, SKIP_ALLOWED)
