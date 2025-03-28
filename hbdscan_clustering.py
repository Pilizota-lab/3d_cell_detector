import os
import numpy as np
from skimage import io, measure, color
import hdbscan
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
import seaborn as sns

# Use the Agg backend for Matplotlib to avoid Tkinter issues
matplotlib.use('Agg')

# Parameters
# DISTANCE_THRESHOLD = 2  # Commenting out the distance threshold for now
MIN_CLUSTER_SIZE = 6  # min cluster size for a cluster to be considered valid
MAX_GAP = 20  # how many times a cell is to reoccur throughout the stack
SKIP_ALLOWED = 1  # cell to be counted as the same cell even if one segmentation mask is missing

# Set min_samples to 2 for HDBSCAN to cluster the same cell across frames
MIN_SAMPLES = 3
def assign_cell_numbers(masks, max_gap, skip_allowed):
    max_label = 0  # tracks highest cell number
    cell_centers = []

    for i, mask in enumerate(masks):  # loop through each mask
        labeled_mask, num_labels = measure.label(mask, return_num=True)  # labels components of the current mask

        for label in range(1, num_labels + 1):  # loop through each labeled object
            object_indices = np.argwhere(labeled_mask == label)  # finds indices of the current label
            center = np.mean(object_indices, axis=0)  # calculates centroid of the object
            if i == 0:  # for the first slice; new label to record the center
                max_label += 1
            else:
                # prev_centers = [c for c in cell_centers if i - c[2] <= max_gap + skip_allowed]  # Commenting out the distance-based tracking
                # if prev_centers:
                #     distances = cdist([center], [c[:2] for c in prev_centers])  # calculate distances to previous centers
                #     min_dist = distances.min()  # find the minimum distance
                #     if min_dist < distance_threshold:  # check if within the distance threshold
                #         min_index = distances.argmin()
                #         existing_label = prev_centers[min_index][3]
                #         cell_centers.append((center[1], center[0], i, existing_label))  # assign existing label
                #     else:
                max_label += 1
                cell_centers.append((center[1], center[0], i, max_label))  # assign new label
                # else:
                #     max_label += 1
                #     cell_centers.append((center[1], center[0], i, max_label))  # assign new label if no previous centers

    return cell_centers

def load_masks(base_directory):
    mask_files = [f for f in os.listdir(base_directory) if f.endswith('.tiff')]
    masks = [io.imread(os.path.join(base_directory, f)) for f in mask_files]
    return masks

def run_cluster_analysis_on_masks(masks, save_directory, min_cluster_size, max_gap, skip_allowed, min_samples):
    cell_centers = assign_cell_numbers(masks, max_gap, skip_allowed)

    Z_SCALING_FACTOR = 5  # Adjust this value based on how much weight you want to give the Z-axis (slice index)
    scaled_centers = np.array([[center[0], center[1], center[2] * Z_SCALING_FACTOR] for center in cell_centers])


    if scaled_centers.size == 0:
        print(f"No centers found in masks for {save_directory}.")
        return 0, [], [], []

    # Perform clustering with min_samples set to 2
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    labels = clusterer.fit_predict(scaled_centers)
    
    # Generate and save condensed tree plot
    condensed_tree_save_path = os.path.join(save_directory, 'condensed_tree.png')
    fig, ax = plt.subplots()
    clusterer.condensed_tree_.plot()
    fig.savefig(condensed_tree_save_path)
    plt.close(fig)
    

    clusterer.condensed_tree_.plot()
    matplotlib.use('Agg')
    
    return len(set(labels) - {-1}), labels, scaled_centers, cell_centers

def visualize_cell_numbers(masks, cell_centers, results_dir, dir_name):
    cell_count_save_path = os.path.join(results_dir, 'cell_counts')
    os.makedirs(cell_count_save_path, exist_ok=True)

    for i, mask in enumerate(masks):
        labeled_mask, _ = measure.label(mask, return_num=True)
        overlay = color.label2rgb(labeled_mask, image=mask, bg_label=0)
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)

        for center in cell_centers:
            if center[2] == i:
                plt.text(center[0], center[1], f'{center[3]}', color='white', fontsize=8, ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.5'))

        plt.title(f'{dir_name} - Slice {i+1} - Cells: {len([c for c in cell_centers if c[2] == i])}')
        plt.axis('off')
        save_path = os.path.join(cell_count_save_path, f'overlay_slice_{i+1}.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Saved overlay to {save_path}")

def plot_3d_clusters(centers, labels, save_path, number_of_clusters):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    scatter = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=colors)

    # Add the number of clusters as text
    ax.text2D(0.05, 0.95, f'Number of Clusters: {number_of_clusters}', transform=ax.transAxes, fontsize=14, verticalalignment='top')

    plt.title('3D Clusters')
    plt.savefig(save_path)
    plt.close(fig)

def estimate_cell_count_range(cell_centers, min_appearance=18, max_appearance=25):
    unique_labels = set([center[3] for center in cell_centers])
    total_labels = len(unique_labels)
    
    estimated_min_cells = round(total_labels / max_appearance)
    estimated_max_cells = round(total_labels / min_appearance)
    
    return estimated_min_cells, estimated_max_cells

def process_directories(base_directory, min_cluster_size, max_gap, skip_allowed, min_samples):
    folder_names = []
    cell_counts = []
    estimated_ranges = []

    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir = os.path.join(dir_path, 'results')
        if os.path.isdir(dir_path) and os.path.exists(results_dir):
            mask_files = sorted([f for f in os.listdir(results_dir) if f.endswith('_label.tiff')],
                                key=lambda x: int(''.join(filter(str.isdigit, x)) or -1))
            masks = [io.imread(os.path.join(results_dir, f)) for f in mask_files]

            number_of_cells, labels, centers, cell_centers = run_cluster_analysis_on_masks(masks, results_dir, min_cluster_size, max_gap, skip_allowed, min_samples)
            print(f"Processed directory '{dir_name}' with {number_of_cells} cells detected.")

            folder_names.append(dir_name)
            cell_counts.append(number_of_cells)

            # Estimate cell count range
            estimated_min_cells, estimated_max_cells = estimate_cell_count_range(cell_centers)
            estimated_ranges.append((estimated_min_cells, estimated_max_cells))

            # Visualize cells with pixel values and centroids
            visualize_cell_numbers(masks, cell_centers, results_dir, dir_name)
            
            # Plot 3D clusters with cell numbers
            plot_3d_clusters(centers, labels, os.path.join(results_dir, '3d_clusters.png'), number_of_cells)

    save_results_to_excel(folder_names, cell_counts, estimated_ranges, os.path.join(base_directory, 'cell_counts_summary.xlsx'))

           
def save_results_to_excel(folder_names, cell_counts, estimated_ranges, output_path):
    results_df = pd.DataFrame({
        'Folder Name': folder_names, 
        'Cell Count': cell_counts, 
        'Estimated Min Cells': [rng[0] for rng in estimated_ranges], 
        'Estimated Max Cells': [rng[1] for rng in estimated_ranges]
    })
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    base_directory = "/home/urte/MEDIAR-main/growth curves test images"
    process_directories(base_directory, MIN_CLUSTER_SIZE, MAX_GAP, SKIP_ALLOWED, MIN_SAMPLES)
