import os
import subprocess
import tempfile
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

def is_image_valid(image_path):
    """Check if the image is valid and does not contain only zero values."""
    try:
        with Image.open(image_path) as img:
            img_array = np.array(img)
            if np.all(img_array == 0):
                return False  # Image contains only zeros
            if img.getbbox() is None:
                return False  # Image is empty
        return True
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return False  # Invalid image file

def run_mediar_prediction(config_path):
    """Executes the MEDIAR prediction using a subprocess within the correct working directory."""
    print(f"Running MEDIAR prediction with config: {config_path}")
    python_exec_path = '/home/urte/miniconda3/envs/mediar/bin/python'
    command = f"{python_exec_path} /home/urte/MEDIAR-main/predict.py --config_path={config_path}"
    subprocess.run(command, shell=True, check=True)

def process_with_cellpose(base_directory):
    """Runs the overlay and 3D projection script with Cellpose."""
    print(f"Running Cellpose processing for directory: {base_directory}")
    python_exec_path = '/home/urte/miniconda3/envs/cellpose/bin/python'
    script_path = '/home/urte/3D modeller/3d_cell_detector/stitch_and_overlay_with_cellpose.py'
    subprocess.run([python_exec_path, script_path])

def run_hdbscan_clustering(base_directory):
    """Runs the HDBSCAN clustering script."""
    print(f"Running HDBSCAN clustering for directory: {base_directory}")
    python_exec_path = '/home/urte/miniconda3/envs/mediar/bin/python'
    script_path = '/home/urte/3D modeller/3d_cell_detector/hdbscan_clustering.py'
    subprocess.run([python_exec_path, script_path])

def main():
    base_directory = "/home/urte/MEDIAR-main/growth curves test images"
    config_template_path = '/home/urte/MEDIAR-main/config/step3_prediction/base_prediction.json'
    log_file_path = os.path.join(base_directory, 'processed_directories_test_mediar.log')

    processed_directories = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as log_file:
            processed_directories.update(log_file.read().splitlines())

    for dir_name in tqdm(
        [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))],
        desc="Processing directories"
    ):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir = os.path.join(dir_path, 'results')

        # Check log only (no longer skipping if 'results' exists)
        if dir_name not in processed_directories:
            print(f"Processing directory: {dir_name}")

            # Load the template config file and update paths
            with open(config_template_path, 'r') as file:
                config = json.load(file)

            config['pred_setups']['input_path'] = dir_path
            config['pred_setups']['output_path'] = results_dir

            # Filter out invalid or zero-valued images
            image_extensions = ['.png', '.jpg', '.jpeg', '.tiff']
            image_files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in image_extensions]

            valid_images = [f for f in image_files if is_image_valid(os.path.join(dir_path, f))]
            if not valid_images:
                print(f"Skipping directory {dir_name}, no valid images found after filtering.")
                continue

            # Save the updated config
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                json.dump(config, tmp)
                tmp_config_path = tmp.name

            try:
                run_mediar_prediction(tmp_config_path)
            except subprocess.CalledProcessError as e:
                print(f"Prediction failed for directory {dir_name} due to {e}, skipping to next directory.")
                os.unlink(tmp_config_path)
                continue

            # Log the processed directory
            with open(log_file_path, 'a') as log_file:
                log_file.write(dir_name + '\n')
            os.unlink(tmp_config_path)

            # Run Cellpose script
            try:
                process_with_cellpose(base_directory)
            except subprocess.CalledProcessError as e:
                print(f"Cellpose processing failed for directory {dir_name} due to {e}, skipping to next directory.")
                continue

            # Run HDBSCAN clustering script
            try:
                run_hdbscan_clustering(base_directory)
            except subprocess.CalledProcessError as e:
                print(f"HDBSCAN clustering failed for directory {dir_name} due to {e}, skipping to next directory.")
                continue

if __name__ == "__main__":
    print("Starting the main pipeline")
    main()
    print("Main pipeline completed")
