import os
import sys
import numpy as np
from PIL import Image

# Configure stdout to flush immediately
sys.stdout.reconfigure(line_buffering=True)

# Directories
image_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/images"
output_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/preprocessed"
os.makedirs(output_dir, exist_ok=True)

def show_stats(image_np, title):
    print(f"{title} - Min: {image_np.min():.4f}, Max: {image_np.max():.4f}, "
          f"Mean: {image_np.mean():.4f}, Std: {image_np.std():.4f}", flush=True)

# According to the paper, these are the preprocessing steps we must apply:
# (P) Clip -> (P) Normalize -> (P) Scale Intensity

def preprocess_image(image_np):
    # Step 1: Clip to [0, 99.5] percentile
    lower = np.percentile(image_np, 0)
    upper = np.percentile(image_np, 99.5)
    clipped = np.clip(image_np, lower, upper)
    show_stats(clipped, "After Clipping")

    # Step 2: Normalize to mean=0, std=1
    mean_val = clipped.mean()
    std_val = clipped.std()
    if std_val < 1e-6:
        # Avoid division by zero if image is constant
        normalized = clipped - mean_val
    else:
        normalized = (clipped - mean_val) / std_val
    show_stats(normalized, "After Normalization (mean=0, std=1)")

    # Step 3: Scale intensity to [0, 1]
    min_val = normalized.min()
    max_val = normalized.max()
    if max_val - min_val < 1e-6:
        # Avoid division by zero if image is flat after normalization
        scaled = np.zeros_like(normalized)
    else:
        scaled = (normalized - min_val) / (max_val - min_val)
    show_stats(scaled, "After Scaling to [0, 1]")

    return scaled

def preprocess_and_save(image_path, output_dir):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load image
    image = Image.open(image_path).convert('L')
    image_np = np.array(image, dtype=np.float32)

    # Show original stats
    show_stats(image_np, f"Original Image ({image_name})")

    # Preprocess the image as per the paper's steps
    processed = preprocess_image(image_np)

    # Convert back to uint8 for saving
    processed_uint8 = (processed * 255).astype(np.uint8)

    preprocessed_image = Image.fromarray(processed_uint8)
    image_save_path = os.path.join(output_dir, f"{image_name}.tiff")
    preprocessed_image.save(image_save_path)
    print(f"Saved preprocessed image to {image_save_path}")

print("Starting preprocessing...", flush=True)

# Process all images
image_files = [f for f in os.listdir(image_dir) if f.endswith('.tiff')]
print(f"Found {len(image_files)} images in {image_dir}", flush=True)

for idx, image_file in enumerate(image_files):
    image_path = os.path.join(image_dir, image_file)
    print(f"\nProcessing {idx+1}/{len(image_files)}: {image_file}", flush=True)
    preprocess_and_save(image_path, output_dir)

print("Preprocessing complete.", flush=True)
