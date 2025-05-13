import os
import numpy as np
from skimage import io, measure, color
import hdbscan
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Physical parameter; calulcated from the graticule
PIXEL_SIZE_X = 0.053125  # µm/px
PIXEL_SIZE_Y = 0.0518519  # µm/px
Z_STEP_SIZE = 1.0  # µm between z-slices
Z_SCALING_FACTOR = 1 / PIXEL_SIZE_X  # 18.82 (1µm z ≈ 18.82 pixels in x)

# Tracking parameters
DISTANCE_THRESHOLD_MICRON = 2.0  # Max allowed 3D movement between frames
MIN_CLUSTER_SIZE = 6             # Min points for valid cluster
MAX_GAP = 15                     # Max frames between reappearances
SKIP_ALLOWED = 1                 # Allowed missing segmentations
MIN_SAMPLES = 3                  # HDBSCAN density control

matplotlib.use('Agg')  # Headless matplotlib

def assign_cell_numbers(masks, max_gap, skip_allowed):
    """Track cells across z-stack with physical unit conversions"""
    max_label = 0
    cell_centers = []

    for z_idx, mask in enumerate(masks):
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        for region in regions:
            y_centroid, x_centroid = region.centroid
            
            # Convert to physical units (µm)
            x_µm = x_centroid * PIXEL_SIZE_X
            y_µm = y_centroid * PIXEL_SIZE_Y
            z_µm = z_idx * Z_STEP_SIZE

            if z_idx == 0:  # First frame initialization
                max_label += 1
                cell_centers.append((x_µm, y_µm, z_µm, z_idx, max_label))
            else:
                # Find candidate matches in previous frames
                prev_candidates = [
                    c for c in cell_centers 
                    if (z_idx - c[3]) <= (max_gap + skip_allowed)
                ]
                
                if prev_candidates:
                    # Create position arrays for distance calculation
                    current_pos = np.array([[x_µm, y_µm, z_µm]])
                    prev_positions = np.array([[c[0], c[1], c[2]] for c in prev_candidates])
                    
                    # Calculate 3D Euclidean distances in µm
                    distances = cdist(current_pos, prev_positions)
                    min_idx = np.argmin(distances)
                    min_dist = distances[0, min_idx]
                    
                    if min_dist <= DISTANCE_THRESHOLD_MICRON:
                        # Assign existing label
                        existing_label = prev_candidates[min_idx][4]
                        cell_centers.append((x_µm, y_µm, z_µm, z_idx, existing_label))
                        continue
                
                # No valid match found - new cell
                max_label += 1
                cell_centers.append((x_µm, y_µm, z_µm, z_idx, max_label))
    
    return cell_centers

def load_masks(base_directory):
    """Load and sort mask files by z-index"""
    mask_files = sorted(
        [f for f in os.listdir(base_directory) if f.endswith('.tiff')],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    return [io.imread(os.path.join(base_directory, f)) for f in mask_files]

def run_cluster_analysis(cell_centers, min_cluster_size):
    """Perform HDBSCAN clustering with z-axis emphasis"""
    if not cell_centers:
        return 0, [], [], []

    # Scale positions for clustering (z-axis emphasized)
    scaled_data = np.array([[
        c[0],  # x in µm
        c[1],  # y in µm 
        c[2] * Z_SCALING_FACTOR  # z scaled to pixel-equivalent units
    ] for c in cell_centers])

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=MIN_SAMPLES,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(scaled_data)
    
    return len(set(labels) - {-1}), labels, scaled_data, cell_centers

def visualize_results(masks, cell_centers, labels, output_dir):
    """Generate diagnostic visualizations"""
    # 3D Cluster Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    
    # Convert back to pixel coordinates for visualization
    x_px = [c[0]/PIXEL_SIZE_X for c in cell_centers]
    y_px = [c[1]/PIXEL_SIZE_Y for c in cell_centers]
    z_idx = [c[3] for c in cell_centers]
    
    ax.scatter(x_px, y_px, z_idx, c=colors)
    ax.set_xlabel('X (px)')
    ax.set_ylabel('Y (px)')
    ax.set_zlabel('Z index')
    plt.savefig(os.path.join(output_dir, '3d_clusters.png'))
    plt.close()

def process_directory(directory_path):
    """Process a single directory of mask files"""
    masks = load_masks(os.path.join(directory_path, 'results'))
    cell_centers = assign_cell_numbers(masks, MAX_GAP, SKIP_ALLOWED)
    n_clusters, labels, scaled_data, raw_centers = run_cluster_analysis(cell_centers, MIN_CLUSTER_SIZE)
    
    # Save results
    output_dir = os.path.join(directory_path, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save cluster data
    pd.DataFrame({
        'x_px': [c[0]/PIXEL_SIZE_X for c in raw_centers],
        'y_px': [c[1]/PIXEL_SIZE_Y for c in raw_centers],
        'z_idx': [c[3] for c in raw_centers],
        'cluster_id': labels
    }).to_csv(os.path.join(output_dir, 'cluster_data.csv'), index=False)
    
    visualize_results(masks, raw_centers, labels, output_dir)
    return n_clusters

if __name__ == "__main__":
    base_dir = "/home/urte/MEDIAR-main/growth curves test images"
    results = []
    
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            try:
                n_cells = process_directory(dir_path)
                results.append({'Folder': dir_name, 'Cell_Count': n_cells})
                print(f"Processed {dir_name}: {n_cells} cells")
            except Exception as e:
                print(f"Error processing {dir_name}: {str(e)}")
    
    # Save final summary
    pd.DataFrame(results).to_excel(
        os.path.join(base_dir, 'cell_counts_summary.xlsx'),
        index=False
    )
