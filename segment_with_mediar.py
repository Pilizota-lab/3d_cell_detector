import os
import json
import subprocess
import tempfile
from tqdm import tqdm
from PIL import Image

def run_mediar_prediction(config_path):
    """Executes the MEDIAR prediction using a subprocess within the correct working directory."""
    python_exec_path = '/home/urte/miniconda3/envs/mediar/bin/python'
    command = f"{python_exec_path} /home/urte/MEDIAR-main/predict.py --config_path={config_path}"
    subprocess.run(command, shell=True, check=True)

def is_image_valid(image_path):
    """Check if the image is valid and not empty."""
    try:
        with Image.open(image_path) as img:
            if img.getbbox() is None:
                return False  # Image is empty
        return True
    except Exception:
        return False  # Invalid image file

base_directory = "/home/urte/MEDIAR-main/test_images/10_15_24"
config_template_path = '/home/urte/MEDIAR-main/config/step3_prediction/base_prediction.json'
log_file_path = os.path.join(base_directory, 'processed_directories_test_mediar.log')

processed_directories = set()
if os.path.exists(log_file_path):
    with open(log_file_path, 'r') as log_file:
        processed_directories.update(log_file.read().splitlines())

for dir_name in tqdm([d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))], desc="Processing directories"):
    dir_path = os.path.join(base_directory, dir_name)
    results_dir = os.path.join(dir_path, 'results')

    # Check if directory has been processed or results already exist
    if dir_name not in processed_directories and not os.path.exists(results_dir):
        print(f"Processing directory: {dir_name}")

        with open(config_template_path, 'r') as file:
            config = json.load(file)

        config['pred_setups']['input_path'] = dir_path
        config['pred_setups']['output_path'] = results_dir

        # Update config for valid image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff']
        image_files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in image_extensions]

        # Skip invalid or empty first image
        if image_files:
            first_image_path = os.path.join(dir_path, image_files[0])
            if not is_image_valid(first_image_path):
                print(f"Skipping first invalid or empty image: {image_files[0]}")
                image_files = image_files[1:]

        # If there are no valid images after filtering, skip this directory
        if not image_files:
            print(f"Skipping directory {dir_name}, no valid images found after filtering.")
            continue

        # Save the updated config with valid images
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
