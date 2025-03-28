from PIL import Image
import numpy as np
import os

def analyze_image_values(input_path):
    # Load the image with Pillow
    with Image.open(input_path) as img:
        # Ensure image is in RGBA mode
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to NumPy array for analysis
        img_array = np.array(img)
        
        # Compute statistics
        for i, color in enumerate(['Red', 'Green', 'Blue', 'Alpha']):
            channel_data = img_array[:,:,i]
            print(f"{color} channel:")
            print(f"  Min: {np.min(channel_data)}")
            print(f"  Max: {np.max(channel_data)}")
            print(f"  Mean: {np.mean(channel_data):.2f}")
            print(f"  Standard Deviation: {np.std(channel_data):.2f}")
            print()

# Example usage:
input_directory = '/home/urte/MEDIAR-main/test_images/ground_truth_for_eval'
# Analyze all PNG images in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):
        input_path = os.path.join(input_directory, filename)
        print(f'Analyzing {filename}')
        analyze_image_values(input_path)
        print('----------------------------------------')

'''
from PIL import Image
import numpy as np
import os

def convert_rgba_to_grayscale_with_alpha(input_path, output_path):
    # Load the image with Pillow
    with Image.open(input_path) as img:
        # Ensure image is in RGBA mode
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Convert to NumPy array to manipulate
        img_array = np.array(img)

        # Calculate the grayscale values using the luminosity method
        grayscale = img_array[:,:,0]*0.299 + img_array[:,:,1]*0.587 + img_array[:,:,2]*0.114

        # Utilize the alpha channel as a mask
        alpha = img_array[:,:,3] / 255.0  # Normalize alpha values to 0-1
        grayscale *= alpha  # Apply alpha mask

        # Enhance mask visibility by thresholding
        threshold = 128  # This is a parameter you might need to adjust
        grayscale = np.where(grayscale > threshold, 255, 0)  # Apply threshold

        # Convert back to uint8
        grayscale = grayscale.astype(np.uint8)

        # Create a new grayscale image
        grayscale_img = Image.fromarray(grayscale, mode='L')
        
        # Save the grayscale image
        grayscale_img.save(output_path, format='TIFF')

# Example usage:
input_directory = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\ground_truth_for_eval'
output_directory = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\test_images\ground_truth_for_eval'

# Convert all PNG images in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.png'):
        input_path = os.path.join(input_directory, filename)
        output_file = filename.replace('.png', '.tiff')
        output_path = os.path.join(output_directory, output_file)
        convert_rgba_to_grayscale_with_alpha(input_path, output_path)
        print(f'Converted {filename} to grayscale TIFF with enhanced visibility.')
'''