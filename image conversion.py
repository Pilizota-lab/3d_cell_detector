import os

def rename_images(directory, base_name="image_", new_name="23_21_31 _", label="_label"):
    # Get all files in the directory
    files = sorted([f for f in os.listdir(directory) if f.startswith(base_name) and f.lower().endswith(('.png', '.tiff', '.jpg', '.jpeg', '.gif', '.bmp'))])
    
    # Loop through the files and rename them
    for i, file in enumerate(files):
        old_path = os.path.join(directory, file)
        extension = os.path.splitext(file)[1]  # Extract the file extension
        new_filename = f"{new_name}{i}{label}{extension}"
        new_path = os.path.join(directory, new_filename)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f'Renamed "{file}" to "{new_filename}"')


# Usage
directory = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\ground_truth_for_eval\ground_truthresults_model_temporal_4\masks'  # Change this to the path of your images folder
rename_images(directory)
