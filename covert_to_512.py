from PIL import Image
import os

def split_image_and_mask(image_path, mask_path, output_directory):
    with Image.open(image_path) as img, Image.open(mask_path) as mask:
        # Calculate the coordinates for two 512x512 crops from the center
        mid_x, mid_y = img.width // 2, img.height // 2
        crop_areas = [
            (mid_x - 512, mid_y - 256, mid_x, mid_y + 256),  # Left crop
            (mid_x, mid_y - 256, mid_x + 512, mid_y + 256)   # Right crop
        ]

        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Crop and save each part for both image and mask
        for i, crop_area in enumerate(crop_areas, start=1):
            cropped_img = img.crop(crop_area)
            cropped_img.save(os.path.join(output_directory, f"{base_name}_part{i}.tif"))

            cropped_mask = mask.crop(crop_area)
            cropped_mask.save(os.path.join(output_directory, f"{base_name}_part{i}_cp_masks.png"))

# Directory containing your 1920x1080 images and masks
source_directory = '/home/urte/3D modeller/3d_cell_detector/trainingdata/Urte_5/Urte_5/'
# Directory where you want to save the cropped 512x512 images and masks
target_directory ='/home/urte/3D modeller/3d_cell_detector/trainingdata/Omni_5/'
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# Process each image in the source directory
for filename in os.listdir(source_directory):
    if filename.lower().endswith('.tiff') and not filename.endswith('_cp_masks.png'):
        image_path = os.path.join(source_directory, filename)
        mask_filename = filename.replace('.tiff', '_cp_masks.png')
        mask_path = os.path.join(source_directory, mask_filename)
        
        if os.path.exists(mask_path):
            split_image_and_mask(image_path, mask_path, target_directory)
        else:
            print(f"Mask file not found for {filename}")

print("Image and mask splitting complete!")
