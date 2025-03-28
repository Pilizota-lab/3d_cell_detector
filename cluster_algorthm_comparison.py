import numpy as np
import hdbscan
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, measure
import seaborn as sns
import os

# Matplotlib settings for 3D plotting
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

# Base directory where the mask files are stored
base_directory = "/home/urte/MEDIAR-main/growth curves test images/24_03_03_23_21_31/results"  # Change this to the path of your directory

# Function to load masks from a specified directory
def load_masks(base_directory):
    mask_files = [f for f in os.listdir(base_directory) if f.endswith('.tiff')]  # Assumes TIFF format
    masks = [io.imread(os.path.join(base_directory, f)) for f in mask_files]
    return masks

# Function to extract centers from masks
def extract_centers_from_masks(masks):
    cell_centers = []
    for i, mask in enumerate(masks):
        labeled_mask, num_labels = measure.label(mask, return_num=True)
        for label in range(1, num_labels + 1):
            object_indices = np.argwhere(labeled_mask == label)
            center = np.mean(object_indices, axis=0)
            cell_centers.append((center[1], center[0], i))  # Swapping x, y for better visualization
    return np.array(cell_centers)

# Load masks from the specified directory
masks = load_masks(base_directory)

# Extract cell centers from masks
data = extract_centers_from_masks(masks)

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Plotting function
def plot_clusters(data, labels, algorithm_name):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=colors, **plot_kwds)
    plt.title(f'Clusters found by {algorithm_name}')
    plt.show()

# Clustering algorithms
def cluster_and_plot(data, algorithm, args, kwargs):
    labels = algorithm(*args, **kwargs).fit_predict(data)
    plot_clusters(data, labels, algorithm.__name__)

# Comparing different algorithms
cluster_and_plot(data_scaled, cluster.KMeans, (), {'n_clusters':50})  # Estimate based on density
cluster_and_plot(data_scaled, cluster.DBSCAN, (), {'eps':0.2, 'min_samples':10})  # Adjusted for dense data
cluster_and_plot(data_scaled, cluster.AgglomerativeClustering, (), {'n_clusters':50, 'linkage':'ward'})
cluster_and_plot(data_scaled, hdbscan.HDBSCAN, (), {'min_clus  ter_size':6, 'min_samples':1})  # Based on provided HDBSCAN params