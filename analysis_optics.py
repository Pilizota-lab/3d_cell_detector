import os
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import support_functions_hdbscan as sf

# Reload your module to ensure the latest changes are applied
importlib.reload(sf)

# Set the necessary environment variables and paths
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
base_directory = r'Z:\Urte\PhD\3D modeller\test_images'
model_identifier = 'temporal_4'
log_file_path = os.path.join(base_directory, 'processed_directories_test_optics.log')
model = sf.load_working_model(r'Z:\Urte\PhD\3D modeller\models\{}'.format(model_identifier))

# Load the list of already processed directories
processed_directories = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_directories.update(log_file.read().splitlines())

# Iterate over each directory and process it
for dir_name in tqdm(os.listdir(base_directory), desc="Processing directories"):
    dir_path = os.path.join(base_directory, dir_name)
    if os.path.isdir(dir_path) and dir_name not in processed_directories:
        print(f"Processing directory: {dir_name}")
        optics_model, number_of_cells = sf.process_directory(dir_path, model, model_identifier)
        print(f"Received OPTICS model: {type(optics_model)} with {number_of_cells} cells.")
        
        # Check for reachability plot and save it with a unique filename
        if hasattr(optics_model, 'reachability_') and hasattr(optics_model, 'ordering_'):
            reachability_plot_path = os.path.join(base_directory, f"{dir_name}_reachability_plot.png")
            sf.plot_reachability(optics_model, reachability_plot_path)
            print(f"Reachability plot saved to: {reachability_plot_path}")
        else:
            print(f"Error: Expected OPTICS model, got {type(optics_model)} instead.")
    else:
        print(f"Skipping already processed directory or file: {dir_name}")


# Assuming collect_and_save_counts is correctly implemented
sf.collect_and_save_counts(base_directory, model_identifier, os.path.join(base_directory, 'test.xlsx'))
