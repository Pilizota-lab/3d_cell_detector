import os
import sys
import numpy as np
from PIL import Image
import random
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

# Configure line buffering for prints
sys.stdout.reconfigure(line_buffering=True)

# User sets this each run:
run_id = 1        # Increment this each time you run to produce aug_1, aug_2, etc.
NUM_AUGS = 5     # How many augmented versions to produce per original
MAX_DISPLAYS = 10 # How many total augmented samples you want to *display*

# Input (preprocessed data)
preprocessed_img_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/preprocessed/images"
preprocessed_lbl_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/preprocessed/labels"

# Output (augmented data)
augmented_img_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/augmented/images"
augmented_lbl_dir = "/home/urte/MEDIAR-main/CellSeg/Labelled/augmented/labels"
os.makedirs(augmented_img_dir, exist_ok=True)
os.makedirs(augmented_lbl_dir, exist_ok=True)

#####################################
# Augmentation probabilities and funcs
#####################################
P_ZOOM = 0.5
P_AXIS_FLIP = 0.5
P_ROTATION = 0.5
P_CELL_AWARE_INTENSITY = 0.25
P_GAUSSIAN_NOISE = 0.25
P_CONTRAST = 0.25
P_GAUSSIAN_SMOOTH = 0.25
P_HIST_SHIFT = 0.25
P_GAUSSIAN_SHARPEN = 0.25

CROP_SIZE = (512, 512)
ZOOM_RANGE = (0.25, 1.5)
GAUSSIAN_NOISE_STD = 0.1
BOUNDARY_VALUE = 255

def boundary_exclusion(label_np):
    label_np[label_np == BOUNDARY_VALUE] = 0
    return label_np

def random_spatial_crop(image_np, label_np):
    h, w = image_np.shape
    ch, cw = CROP_SIZE
    if h < ch or w < cw:
        pad_h = max(ch - h, 0)
        pad_w = max(cw - w, 0)
        image_np = np.pad(image_np, ((0,pad_h),(0,pad_w)), mode='reflect')
        label_np = np.pad(label_np, ((0,pad_h),(0,pad_w)), mode='reflect')
        h, w = image_np.shape

    top = random.randint(0, h - ch)
    left = random.randint(0, w - cw)
    image_crop = image_np[top:top+ch, left:left+cw]
    label_crop = label_np[top:top+ch, left:left+cw]
    return image_crop, label_crop

def random_zoom(image_np, label_np):
    scale = random.uniform(ZOOM_RANGE[0], ZOOM_RANGE[1])
    h, w = image_np.shape
    new_h = int(h*scale)
    new_w = int(w*scale)

    img_pil = Image.fromarray((image_np*255).astype(np.uint8))
    img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
    image_zoom = np.array(img_pil, dtype=np.float32)/255.0

    lbl_pil = Image.fromarray(label_np)
    lbl_pil = lbl_pil.resize((new_w, new_h), Image.NEAREST)
    label_zoom = np.array(lbl_pil, dtype=np.uint8)
    return image_zoom, label_zoom

def axis_flip(image_np, label_np):
    image_flip = np.flip(image_np, axis=1)
    label_flip = np.flip(label_np, axis=1)
    return image_flip, label_flip

def random_rotation(image_np, label_np):
    k = random.choice([1,2,3])  # rotate by 90,180,270
    image_rot = np.rot90(image_np, k)
    label_rot = np.rot90(label_np, k)
    return image_rot, label_rot

def cell_aware_intensity(image_np):
    scale_factor = random.uniform(1.0, 1.7)
    scaled = image_np * scale_factor
    return np.clip(scaled, 0, 1)

def add_gaussian_noise(image_np):
    noise = np.random.normal(0, GAUSSIAN_NOISE_STD, image_np.shape)
    noised = image_np + noise
    return np.clip(noised, 0, 1)

def adjust_contrast(image_np):
    gamma = random.uniform(0.0, 2.0)
    mean_val = image_np.mean()
    adjusted = ((image_np - mean_val)**gamma) + mean_val
    adjusted = np.clip(adjusted, 0, 1)
    return adjusted

def gaussian_smooth(image_np):
    smoothed = gaussian_filter(image_np, sigma=1.0)
    return np.clip(smoothed, 0, 1)

def histogram_shift(image_np):
    shift_val = random.uniform(-0.1, 0.1)
    shifted = image_np + shift_val
    shifted = np.clip(shifted, 0, 1)
    return shifted

def gaussian_sharpen(image_np):
    sigma = random.choice([0.5, 1.0])
    alpha = random.uniform(10.0, 30.0)
    blurred = gaussian_filter(image_np, sigma=sigma)
    sharpened = image_np + alpha * (image_np - blurred)
    return np.clip(sharpened, 0, 1)

def apply_augmentations(image_np, label_np):
    """
    Apply multiple augmentations in a single pass.
    Each augmentation is randomly applied based on
    the probability constants above.
    """
    # Spatial Aug
    if random.random() < P_ZOOM:
        image_np, label_np = random_zoom(image_np, label_np)
    image_np, label_np = random_spatial_crop(image_np, label_np)
    if random.random() < P_AXIS_FLIP:
        image_np, label_np = axis_flip(image_np, label_np)
    if random.random() < P_ROTATION:
        image_np, label_np = random_rotation(image_np, label_np)

    # Boundary Exclusion
    label_np = boundary_exclusion(label_np)

    # Intensity Aug
    if random.random() < P_CELL_AWARE_INTENSITY:
        image_np = cell_aware_intensity(image_np)
    if random.random() < P_GAUSSIAN_NOISE:
        image_np = add_gaussian_noise(image_np)
    if random.random() < P_CONTRAST:
        image_np = adjust_contrast(image_np)
    if random.random() < P_GAUSSIAN_SMOOTH:
        image_np = gaussian_smooth(image_np)
    if random.random() < P_HIST_SHIFT:
        image_np = histogram_shift(image_np)
    if random.random() < P_GAUSSIAN_SHARPEN:
        image_np = gaussian_sharpen(image_np)

    return image_np, label_np

def save_image_and_label(image_np, label_np, name_prefix):
    img_uint8 = (image_np * 255).astype(np.uint8)
    lbl_uint8 = label_np.astype(np.uint8)
    Image.fromarray(img_uint8).save(os.path.join(augmented_img_dir, f"{name_prefix}.tiff"))
    Image.fromarray(lbl_uint8).save(os.path.join(augmented_lbl_dir, f"{name_prefix}.tiff"))

def display_image_and_label(image_np, label_np, title=""):
    """
    Display image and label side by side for visual confirmation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(image_np, cmap='gray')
    axes[0].set_title(f"{title} - Image")
    axes[0].axis("off")
    
    axes[1].imshow(label_np, cmap='gray')
    axes[1].set_title(f"{title} - Label")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# Main script
# -------------------------------------------------------------------------
image_files = [f for f in os.listdir(preprocessed_img_dir) if f.endswith('.tiff')]
pairs = []
for img_file in image_files:
    img_name = os.path.splitext(img_file)[0]
    label_candidate_1 = os.path.join(preprocessed_lbl_dir, img_name + "_label.tiff")
    label_candidate_2 = os.path.join(preprocessed_lbl_dir, img_name + "__label.tiff")
    if os.path.exists(label_candidate_1):
        pairs.append((img_file, img_name + "_label.tiff"))
    elif os.path.exists(label_candidate_2):
        pairs.append((img_file, img_name + "__label.tiff"))
    else:
        continue

print(f"Found {len(pairs)} image-label pairs.")

# We'll store the (image_np, label_np, prefix) for display after everything is processed.
all_for_display = []

for idx, (img_file, lbl_file) in enumerate(pairs):
    img_path = os.path.join(preprocessed_img_dir, img_file)
    lbl_path = os.path.join(preprocessed_lbl_dir, lbl_file)
    image = Image.open(img_path).convert('L')
    label = Image.open(lbl_path).convert('L')

    image_np = np.array(image, dtype=np.float32) / 255.0
    label_np = np.array(label, dtype=np.uint8)

    base_name = os.path.splitext(img_file)[0]
    
    # -- Save original
    orig_prefix = f"aug_{run_id}_{base_name}_original"
    save_image_and_label(image_np, label_np, orig_prefix)
    # Also store for later possible display
    all_for_display.append((image_np, label_np, orig_prefix))

    # -- Generate multiple augmented versions
    for aug_i in range(NUM_AUGS):
        aug_img, aug_lbl = apply_augmentations(image_np, label_np)
        aug_prefix = f"aug_{run_id}_{base_name}_aug_{aug_i+1}"
        save_image_and_label(aug_img, aug_lbl, aug_prefix)
        # Store for later display
        all_for_display.append((aug_img, aug_lbl, aug_prefix))

# Randomly select up to MAX_DISPLAYS for actual plotting
print(f"Total stored samples for display: {len(all_for_display)}")
to_show = random.sample(all_for_display, min(MAX_DISPLAYS, len(all_for_display)))

# Display them
for (img_np, lbl_np, prefix) in to_show:
    display_image_and_label(img_np, lbl_np, prefix)

print("Done. Original and augmented images have been saved with `aug_{run_id}_` prefix.")
