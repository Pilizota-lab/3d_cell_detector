import os
from PIL import Image

def convert_png_to_tiff(source_dir, output_dir):
    # Check if the output directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith("_label.png") or filename.endswith("cp_masks.png"):  # Check for specific PNG files
            # Create the full file path
            file_path = os.path.join(source_dir, filename)
            
            # Open the image file
            with Image.open(file_path) as img:
                # Convert the image to grayscale if it's not already
                if img.mode != "L":
                    img = img.convert("L")  # Convert to grayscale (mode "L")
                
                # Define the output path for the TIFF file
                base_name = filename.replace("_label.png", "").replace("cp_masks.png", "")
                output_path = os.path.join(output_dir, f"{base_name}_label.tiff")
                
                # Save the image as TIFF with the desired settings
                img.save(output_path, format='TIFF', compression='none')

    print("Conversion complete.")

# Specify the source and output directories
source_directory = '/home/urte/MEDIAR-main/CellSeg/Labelled/labels'
output_directory = '/home/urte/MEDIAR-main/CellSeg/Labelled/labels'

# Call the function with the specified paths
convert_png_to_tiff(source_directory, output_directory)

