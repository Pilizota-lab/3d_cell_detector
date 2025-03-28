import os
import json
import importlib
import subprocess
import tempfile
import Support_functions_mediar as sf
from tqdm import tqdm

def run_mediar_prediction(config_path):
    """Executes the MEDIAR prediction using a subprocess within the correct working directory."""
    python_exec_path = '/home/urte/miniconda3/envs/mediar/bin/python'
    command = f"{python_exec_path} /home/urte/MEDIAR-main/predict.py --config_path={config_path}"
    subprocess.run(command, shell=True, check=True)

base_directory = "/home/urte/MEDIAR-main/test_images/08_12_23"
config_template_path = '/home/urte/MEDIAR-main/config/step3_prediction/base_prediction.json'
log_file_path = os.path.join(base_directory, 'processed_directories_test_mediar.log')
output_file = os.path.join(base_directory, 'cell_counts_summary.xlsx')

folder_names = []
cell_counts = []

processed_directories = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_directories.update(log_file.read().splitlines())

for dir_name in tqdm([d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))], desc="Processing directories"):
    dir_path = os.path.join(base_directory, dir_name)
    results_dir = os.path.join(dir_path, 'results')
    if dir_name not in processed_directories and not os.path.exists(results_dir):
        print(f"Processing directory: {dir_name}")

        # Create a temporary JSON configuration specific for the current directory
        with open(config_template_path, 'r') as file:
            config = json.load(file)
        
        config['pred_setups']['input_path'] = dir_path
        config['pred_setups']['output_path'] = results_dir

        # Use a temporary file to avoid any conflicts or need for cleanup
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            json.dump(config, tmp)
            tmp_config_path = tmp.name
        
        try:
            run_mediar_prediction(tmp_config_path)
        except subprocess.CalledProcessError as e:
            print(f"Prediction failed for directory {dir_name} due to {e}, skipping to next directory.")
            os.unlink(tmp_config_path)  # Clean up the temporary config file after use
            continue  # Skip to the next directory if prediction fails

        try:
            number_of_cells = sf.process_directory(dir_path, config)
            folder_names.append(dir_name)
            cell_counts.append(number_of_cells)
            print(f"Processed directory '{dir_name}' with {number_of_cells} cells detected.")
            with open(log_file_path, 'a') as log_file:
                log_file.write(dir_name + '\n')
        except Exception as e:
            print(f"Failed to process directory {dir_name} due to {e}")
        
        os.unlink(tmp_config_path)  # Clean up the temporary config file after use

if folder_names and cell_counts:
    sf.save_results_to_excel(folder_names, cell_counts, output_file)
else:
    print("No results to save.")
