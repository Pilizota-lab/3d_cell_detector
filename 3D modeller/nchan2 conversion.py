import cv2
import os
import numpy as np 

# Define the input and output folders
input_folder = '/home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5'  # Replace with the path to your input folder
output_folder = '/home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5/nchan2'  # Replace with the path to your output folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loop through each file in the folder
for file in files:
    # Check if the file is an image (you can specify image extensions)
    if file.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
        # Load the RGB image
        rgb_image = cv2.imread(os.path.join(input_folder, file))
        
        # Split the RGB channels into two separate channels (Red and Green)
        red_channel = rgb_image[:, :, 0]  # Red channel
        green_channel = rgb_image[:, :, 1]  # Green channel

        # Create a blank blue channel (or replace with desired channel)
        blue_channel = np.zeros_like(red_channel)

        # Stack the channels to create a two-channel image
        two_channel_image = np.dstack((red_channel, green_channel, blue_channel))

        # Save the two-channel image to the output folder with the same filename
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, two_channel_image)
