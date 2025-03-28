from PIL import Image
import os

# Directory containing your 1920x1080 images
source_directory = '//home/urte/3D modeller/3d_cell_detector/trainingdata/Urte_5/Urte_5/'
# Directory where you want to save the resized 512x512 images
target_directory = '//home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5/'

# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Iterate over all files in the source directory
for filename in os.listdir(source_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        # Open the image
        img_path = os.path.join(source_directory, filename)
        with Image.open(img_path) as img:
            # Resize the image
            img_resized = img.resize((512, 512), Image.ANTIALIAS)

            # Save the resized image to the target directory
            img_resized.save(os.path.join(target_directory, filename))

print("Resizing complete!")
