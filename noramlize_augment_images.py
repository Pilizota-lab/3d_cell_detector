import torchvision.transforms.functional as TF
import random
from PIL import Image
import os
import glob

# Set up the directory paths
input_image_directory = '/home/urte/MEDIAR-main/Data augmentation/images'  # Directory containing input TIFF files
input_label_directory = '/home/urte/MEDIAR-main/Data augmentation/Labels'  # Directory containing label TIFF files
output_image_directory = '/home/urte/MEDIAR-main/Data augmentation/images/augmented'  # Directory to save augmented images
output_label_directory ='/home/urte/MEDIAR-main/Data augmentation/Labels/augmented'  # Directory to save augmented labels

# Ensure output directories exist
if not os.path.exists(output_image_directory):
    os.makedirs(output_image_directory)
if not os.path.exists(output_label_directory):
    os.makedirs(output_label_directory)

# Iterate over TIFF files and apply augmentation and normalization
for image_path in glob.glob(os.path.join(input_image_directory, '*.tiff')):
    label_path = os.path.join(input_label_directory, os.path.basename(image_path).replace('.tiff', '_label.tiff'))
    if not os.path.exists(label_path):
        label_path = os.path.join(input_label_directory, os.path.basename(image_path).replace('.tiff', '__label.tiff'))

    # Open image and label
    image = Image.open(image_path).convert('L')  # Open the image and convert to grayscale if needed
    label = Image.open(label_path).convert('L')  # Open the corresponding label

    # Apply the same transformations to both image and label
    # Random horizontal flip
    if random.random() > 0.5:
        image = TF.hflip(image)
        label = TF.hflip(label)

    # Random vertical flip
    if random.random() > 0.5:
        image = TF.vflip(image)
        label = TF.vflip(label)

    # Random rotation
    angle = random.uniform(0, 360)
    image = TF.rotate(image, angle)
    label = TF.rotate(label, angle)

    # Random crop and resize
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=(0.25, 1.5), ratio=(1.0, 1.0))
    image = TF.resized_crop(image, i, j, h, w, size=(512, 512))
    label = TF.resized_crop(label, i, j, h, w, size=(512, 512))

    # Random brightness and contrast changes
    image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.7, 1.3))
    image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.7, 1.3))

    # Gaussian blur (only applied to the image, not the label)
    if random.random() > 0.75:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))

    # Convert image to tensor and normalize
    image = TF.to_tensor(image)
    image = TF.normalize(image, mean=[0.5], std=[0.5])

    # Convert label to tensor (without normalization)
    label = TF.to_tensor(label)

    # Save the augmented image and label
    image_filename = os.path.basename(image_path).replace('.tiff', '_augmented.tiff')
    label_filename = os.path.basename(label_path).replace('.tiff', '_augmented.tiff')
    TF.to_pil_image(image).save(os.path.join(output_image_directory, image_filename))
    TF.to_pil_image(label).save(os.path.join(output_label_directory, label_filename))

print("Augmented images and labels saved to", output_image_directory, "and", output_label_directory)
