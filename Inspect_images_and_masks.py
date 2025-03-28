import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import random

# Define paths
image_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/augmented/images"
label_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/augmented/labels"
output_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Standardize label naming to "_label.tiff"
def standardize_label_naming(label_dir):
    """Renames labels with '__label.tiff' suffix to '_label.tiff' for consistency."""
    for label_file in os.listdir(label_dir):
        if label_file.endswith("__label.tiff"):
            new_name = label_file.replace("__label.tiff", "_label.tiff")
            os.rename(os.path.join(label_dir, label_file), os.path.join(label_dir, new_name))
            print(f"Renamed {label_file} to {new_name}")

# Pre-processing steps with pixel value logging
def clip_image(image_np):
    lower_bound = np.percentile(image_np, 0)
    upper_bound = np.percentile(image_np, 99.5)
    clipped = np.clip(image_np, lower_bound, upper_bound)
    display_step(clipped, "Clipped Image", np.min(clipped), np.max(clipped))
    return clipped

def normalize_image(image_np):
    normalized = (image_np - np.mean(image_np)) / np.std(image_np)
    display_step(normalized, "Normalized Image", np.min(normalized), np.max(normalized))
    return normalized

def scale_intensity(image_np):
    scaled = np.clip(image_np / 255.0, 0, 1)
    display_step(scaled, "Scaled Intensity Image", np.min(scaled), np.max(scaled))
    return scaled

# Augmentation functions with pixel value logging
def zoom(image, scale_range=(0.25, 1.5)):
    scale = random.uniform(*scale_range)
    new_size = (int(image.width * scale), int(image.height * scale))
    image_resized = image.resize(new_size, Image.NEAREST)
    display_step(np.array(image_resized), "Zoomed Image", np.min(image_resized), np.max(image_resized))
    return image_resized

def spatial_crop(image, crop_size=(512, 512)):
    width, height = image.size
    if width < crop_size[0] or height < crop_size[1]:
        image = image.resize(crop_size)
    else:
        left = random.randint(0, width - crop_size[0])
        top = random.randint(0, height - crop_size[1])
        crop_box = (left, top, left + crop_size[0], top + crop_size[1])
        image = image.crop(crop_box)
    display_step(np.array(image), "Cropped Image", np.min(image), np.max(image))
    return image

def axis_flip(image):
    flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
    display_step(np.array(flipped), "Flipped Image", np.min(flipped), np.max(flipped))
    return flipped

def rotate(image):
    angle = random.choice([90, 180, 270])
    rotated = image.rotate(angle)
    display_step(np.array(rotated), f"Rotated Image {angle}°", np.min(rotated), np.max(rotated))
    return rotated

def cell_aware_intensity(image_np, intensity_range=(1.0, 1.7)):
    scale = random.uniform(*intensity_range)
    adjusted = np.clip(image_np * scale, 0, 1)
    display_step(adjusted, "Cell-Aware Intensity Image", np.min(adjusted), np.max(adjusted))
    return adjusted

def gaussian_noise(image_np, mean=0, std=0.1):
    noise = np.random.normal(mean, std, image_np.shape)
    noisy = np.clip(image_np + noise, 0, 1)
    display_step(noisy, "Gaussian Noise Image", np.min(noisy), np.max(noisy))
    return noisy

def contrast_adjust(image_np, gamma_range=(0.0, 2.0)):
    gamma = random.uniform(*gamma_range)
    adjusted = np.clip(image_np ** gamma, 0, 1)
    display_step(adjusted, "Contrast Adjusted Image", np.min(adjusted), np.max(adjusted))
    return adjusted

def gaussian_smoothing(image):
    smoothed = image.filter(ImageFilter.GaussianBlur(radius=1.0))
    display_step(np.array(smoothed), "Gaussian Smoothed Image", np.min(smoothed), np.max(smoothed))
    return smoothed

def histogram_shift(image):
    enhancer = ImageEnhance.Brightness(image)
    shifted = enhancer.enhance(random.uniform(0.8, 1.2))
    display_step(np.array(shifted), "Histogram Shifted Image", np.min(shifted), np.max(shifted))
    return shifted

def gaussian_sharpening(image, alpha_range=(10, 30)):
    alpha = int(random.uniform(*alpha_range))
    sharpened = image.filter(ImageFilter.UnsharpMask(radius=1.0, percent=alpha))
    display_step(np.array(sharpened), "Gaussian Sharpened Image", np.min(sharpened), np.max(sharpened))
    return sharpened

def boundary_exclusion(label_np):
    excluded = np.where(label_np == 1, 0, label_np)
    display_step(excluded, "Boundary Excluded Label", np.min(excluded), np.max(excluded))
    return excluded

# Display helper function
def display_step(image_np, title, min_val, max_val):
    print(f"{title} - Min: {min_val}, Max: {max_val}")
    plt.figure()
    plt.imshow(image_np, cmap='gray')
    plt.title(f"{title} (Min: {min_val}, Max: {max_val})")
    plt.axis('off')
    plt.show()

# Apply preprocessing and save each augmentation separately
def preprocess_and_save(image_path, label_path, output_dir, idx):
    image = Image.open(image_path).convert('L')
    label = Image.open(label_path).convert('L')

    # Pre-processing steps
    image_np = np.array(image, dtype=np.float32)
    label_np = np.array(label, dtype=np.float32)

    image_np = clip_image(image_np)
    image_np = normalize_image(image_np)
    image_np = scale_intensity(image_np)

    # Convert back to PIL for augmentations
    base_image = Image.fromarray((image_np * 255).astype(np.uint8))
    base_label = Image.fromarray(label_np.astype(np.uint8))

    # Save each augmentation independently
    augmentations = [
        ("zoom", lambda img: zoom(img)),
        ("crop", lambda img: spatial_crop(img)),
        ("flip", lambda img: axis_flip(img)),
        ("rotate", lambda img: rotate(img)),
        ("cell_intensity", lambda img_np: Image.fromarray((cell_aware_intensity(np.array(img_np) / 255.0) * 255).astype(np.uint8))),
        ("gaussian_noise", lambda img_np: Image.fromarray((gaussian_noise(np.array(img_np) / 255.0) * 255).astype(np.uint8))),
        ("contrast", lambda img_np: Image.fromarray((contrast_adjust(np.array(img_np) / 255.0) * 255).astype(np.uint8))),
        ("smoothing", lambda img: gaussian_smoothing(img)),
        ("histogram_shift", lambda img: histogram_shift(img)),
        ("sharpening", lambda img: gaussian_sharpening(img))
    ]

    for aug_name, aug_func in augmentations:
        if "np" in aug_name:
            aug_image = aug_func(base_image)
        else:
            aug_image = aug_func(base_image)
        
        # Save augmented image and corresponding label
        aug_image.save(os.path.join(output_dir, f"{aug_name}_image_{idx}.tiff"))
        base_label.save(os.path.join(output_dir, f"{aug_name}_label_{idx}.tiff"))

# Standardize label names, preprocess, and augment images
standardize_label_naming(label_dir)

# Process all images in directory
for idx, image_file in enumerate(os.listdir(image_dir)):
    if image_file.endswith('.tiff'):
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, image_file.replace('.tiff', '_label.tiff'))
        if os.path.exists(label_path):
            preprocess_and_save(image_path, label_path, output_dir, idx)
        else:
            print(f"No corresponding label found for {image_file}")
