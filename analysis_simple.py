import os
import importlib
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import support_functions as sf

# Reload your module to ensure the latest changes are applied
importlib.reload(sf)

# SSet the necessary environment variables and paths
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
base_directory = "/home/urte/MEDIAR-main/growth curves test images"
model_identifier = 'temporal_5'
log_file_path = os.path.join(base_directory, 'processed_directories_temporal_5.log')
model = sf.load_working_model(r'/mnt/z/Urte/PhD/3D modeller/models/{}'.format(model_identifier))

# Prepare lists to store results
folder_names = []
cell_counts = []

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
        number_of_cells = sf.process_directory(dir_path, model, model_identifier)
        folder_names.append(dir_name)
        cell_counts.append(number_of_cells)
        print(f"Received number of cells: {number_of_cells} from {dir_name}")
        sf.update_processed_directories_log(dir_name, log_file_path)
    else:
        print(f"Skipping already processed directory or file: {dir_name}")

# Save the results to an Excel file if data was collected
output_file = os.path.join(base_directory, 'cell_counts_summary_temp_5.xlsx')
if folder_names and cell_counts:
    sf.save_results_to_excel(folder_names, cell_counts, output_file)
else:
    print("No results to save.")
