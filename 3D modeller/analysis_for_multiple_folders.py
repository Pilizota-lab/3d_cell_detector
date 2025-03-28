import os
import support_functions as sf
from tqdm import tqdm
import importlib
importlib.reload(sf)


# Set the base directory, model identifier, and log file path
base_directory = r'X:\phD\captures\mg1655\06_02_24'
model_identifier = 'temporal_4'
log_file_path = os.path.join(base_directory, 'processed_directories_temporal_4.log')
model = sf.load_working_model(r'Z:\Urte\PhD\3D modeller\models\{}'.format(model_identifier))

folder_names = []
cell_counts = []

# Load processed directories from the log file
processed_directories = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_directories.update(log_file.read().splitlines())

# Process each directory in the base directory with a progress bar
for dir_name in tqdm(os.listdir(base_directory), desc="Processing directories"):
    dir_path = os.path.join(base_directory, dir_name)
    if os.path.isdir(dir_path) and dir_name not in processed_directories:
        print(f"Processing directory: {dir_name}")
        count = sf.process_directory(dir_path, model, model_identifier)
        folder_names.append(dir_name)
        cell_counts.append(count)

        # Update the log file
        sf.update_processed_directories_log(dir_name, log_file_path)
    else:
        print(f"Skipping already processed directory: {dir_name}")

# Save the results to an Excel file
output_file = os.path.join(base_directory,'06_02_24_model_temporal_4.xlsx')
#sf.save_results_to_excel(folder_names, cell_counts, output_file)
sf.collect_and_save_counts(base_directory, model_identifier, output_file)

