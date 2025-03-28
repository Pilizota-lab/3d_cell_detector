import cv2
import os
import numpy as np

# Define the input and output folders
input_folder = '/home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5'
output_folder = '/home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5/nchan2'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the input folder
files = os.listdir(input_folder)

# Loop through each file in the folder
for file in files:
    if file.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
        rgb_image = cv2.imread(os.path.join(input_folder, file))

        if rgb_image.shape[2] == 3:
            grayscale_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            two_channel_image = np.dstack((grayscale_image, grayscale_image))

            # Append a dummy channel to create a 3-channel image
            dummy_channel = np.zeros_like(grayscale_image)
            three_channel_image = np.dstack((two_channel_image, dummy_channel))

            # Save the image as PNG
            output_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.png')
            cv2.imwrite(output_path, three_channel_image)
