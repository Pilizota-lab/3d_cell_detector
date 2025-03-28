from cellpose import models
from cellpose.io import imread
from skimage import io, measure, color, segmentation, transform
import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import re
import hdbscan
from mpl_toolkits.mplot3d import Axes3D


def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_working_model(path_to_model):
    # Load and Run Cellpose Model
    model = models.CellposeModel(pretrained_model=path_to_model, gpu=True)
    return model


def update_processed_directories_log(directory, log_file_path):
    """Update the log file with the processed directory."""
    with open(log_file_path, 'a') as log_file:
        log_file.write(directory + '\n')
    print(f"Directory {directory} added to log.")


def process_directory(directory, model, model_identifier, cellpose_diameter=30, hdbscan_min_cluster_size=5,
                      hdbscan_min_samples=None, image_resize_factor=3):
    """
    Processes a directory of images using the Cellpose model for cell detection and HDBSCAN for clustering analysis. 
    Saves masks, overlays, and performs 3D clustering analysis on the detected cells.
    
    :param directory: String. Path to the directory containing images to be processed.
    :param model: CellposeModel object. Loaded Cellpose model to be used for cell detection.
    :param model_identifier: String. Identifier for the model, used in naming output directories.
    :param cellpose_diameter: Float or None. Diameter of the cells to be detected. If None, the Cellpose model automatically estimates it.
    :param hdbscan_min_cluster_size: Int. The minimum size of clusters in HDBSCAN.
    :param hdbscan_min_samples: Int or None. The 'min_samples' parameter for HDBSCAN, indicating the number of samples in a neighborhood for a point to be considered a core point.
    :param image_resize_factor: Int. Factor by which images are resized before processing. Helps in reducing computation time and memory usage.
    :return: Tuple. HDBSCAN model and total number of unique cells detected and clustered across all images in the directory.
    """
    images = load_images(directory, image_resize_factor)

    # Create a directory for results inside the current directory
    results_dir = os.path.join(directory, f'results_model{model_identifier}_HDBSCAN')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Run cellpose model with the specified diameter
    channels = [[0, 0]]  # Assuming grayscale images
    masks, flows, styles = model.eval(images, diameter=cellpose_diameter, channels=channels, compute_masks=True)

    # Save Masks, Overlays, and create 3D Stack
    save_masks_overlay(images, masks, results_dir)
    create_3d_stack(masks, results_dir)

    # Count Objects and Save the Count using HDBSCAN parameters
    number_of_cells, hdbscan_model = run_cluster_analysis_on_masks(masks, results_dir,
                                                                   hdbscan_min_cluster_size=hdbscan_min_cluster_size,
                                                                   hdbscan_min_samples=hdbscan_min_samples)

    count_filename = os.path.join(results_dir, 'cell_count.txt')
    with open(count_filename, 'w') as file:
        file.write(str(number_of_cells))

    return hdbscan_model, number_of_cells


def load_images(path_to_images, image_resize_factor=3):
    img_dir = path_to_images
    image_files = [f for f in os.listdir(img_dir) if f.endswith('.tiff')]
    sorted_image_files = sorted(image_files, key=numerical_sort_key)

    images = []
    for image_file in sorted_image_files:
        image_path = os.path.join(img_dir, image_file)
        image = io.imread(image_path)

        # Resize the image
        new_height = image.shape[0] // image_resize_factor
        new_width = image.shape[1] // image_resize_factor

        shrunken_image = transform.resize(image, (new_height, new_width), anti_aliasing=True)
        images.append(shrunken_image)
    return images


def save_masks_overlay(images, masks, save_directory):
    # Ensure the directories for saving masks and overlays exist
    masks_dir = os.path.join(save_directory, 'masks')
    overlay_dir = os.path.join(save_directory, 'overlay')
    Path(masks_dir).mkdir(parents=True, exist_ok=True)
    Path(overlay_dir).mkdir(parents=True, exist_ok=True)

    # Save masks and overlays
    for i, (image, mask) in enumerate(zip(images, masks)):
        mask_path = os.path.join(masks_dir, f'image_{i}.png')
        overlay_path = os.path.join(overlay_dir, f'image_{i}.png')

        plt.imsave(mask_path, mask)
        labeled_image = color.label2rgb(mask, image, alpha=0.2)
        plt.imsave(overlay_path, labeled_image)

    return [color.label2rgb(mask, image, alpha=0.2) for image, mask in zip(images, masks)]


def create_3d_stack(masks, save_directory):
    from pathlib import Path
    from skimage import measure
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    stack_dir = os.path.join(save_directory, "3d_images")
    Path(stack_dir).mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            z_coords = np.full((object_indices.shape[0],), i)
            color = np.random.rand(3,).tolist()  # Convert numpy array to list
            ax.scatter(object_indices[:, 1], object_indices[:, 0], z_coords, color=color, marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Slice (Object Index)')
    plt.savefig(os.path.join(stack_dir, "3d_stack.png"), dpi=300)


def run_cluster_analysis_on_masks(masks, save_directory, hdbscan_min_cluster_size=5, hdbscan_min_samples=None,
                                  do_plot=True):
    centers = []
    for i, mask in enumerate(masks):
        labeled_mask = measure.label(mask > 0, connectivity=mask.ndim)
        for region in measure.regionprops(labeled_mask):
            center_y, center_x = region.centroid
            centers.append([center_x, center_y, i])

    data_array = np.array(centers)
    cluster = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size, min_samples=hdbscan_min_samples).fit(data_array)
    labels = cluster.labels_
    labelset = set(labels)

    # Count unique clusters excluding noise (-1)
    if -1 in labelset:
        labelset.remove(-1)
    total_objects = len(labelset)

    # Optional: 3D plotting
    if do_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = set(labels) - {-1}  # Exclude noise
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for k in unique_labels:
            class_member_mask = (labels == k)
            xyz = data_array[class_member_mask]
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=plt.cm.Spectral(k / max(labels)), label=f'Cluster {k}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Clustered Objects: {total_objects} Clusters')
        scatter_plot_path = os.path.join(save_directory, "3d_images/clustered_objects_3d_plot.png")
        plt.savefig(scatter_plot_path)
        plt.show()

    count_filename = os.path.join(save_directory, 'total_clusters_count.txt')
    with open(count_filename, 'w') as count_file:
        count_file.write(str(total_objects))

    return total_objects, cluster


def save_results_to_excel(folder_names, cell_counts, output_path):
    """Save cell count results to an Excel file."""
    results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
    results_df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


def collect_and_save_counts(base_directory, model_identifier, output_file):
    folder_names = []
    cell_counts = []

    # Iterate through each directory in the base directory
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir_name = f'results_model{model_identifier}'
        results_dir_path = os.path.join(dir_path, results_dir_name)
        count_file_path = os.path.join(results_dir_path, 'cell_count.txt')

        # Check if the cell count file exists
        if os.path.isdir(dir_path) and os.path.exists(count_file_path):
            with open(count_file_path, 'r') as count_file:
                count = count_file.read().strip()
                folder_names.append(dir_name)
                cell_counts.append(int(count))

    # Save the results to an Excel file
    if folder_names and cell_counts:
        results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")
