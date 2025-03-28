import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage import io, img_as_ubyte

# Configuration paths
image_dir = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\CellSeg\Labelled\images'
mask_dir = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\CellSeg\Labelled\labels'
output_dir = r'\\wsl.localhost\Ubuntu\home\urte\MEDIAR-main\CellSeg\Labelled\labels'  # For saving edited masks

# Ensuring the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Global variables for interaction
drawing = False
value = 255  # Drawing in white (mask present)

def toggle_value(event):
    global value
    value = 0 if value == 255 else 255

def save_mask(event, mask, mask_path):
    save_path = os.path.join(output_dir, os.path.basename(mask_path))
    io.imsave(save_path, img_as_ubyte(mask))
    print(f"Mask saved to {save_path}")

def onclick(event):
    global drawing
    if event.button == 1 and event.inaxes:
        drawing = True
        onmotion(event)  # Call onmotion to draw on the initial click

def onrelease(event):
    global drawing
    drawing = False

def onmotion(event):
    if drawing and event.inaxes:
        x, y = int(event.xdata), int(event.ydata)
        mask[y, x] = value
        mask_overlay.set_data(mask)
        fig.canvas.draw_idle()

def edit_mask(image_path, mask_path):
    global fig, mask_overlay, mask

    image = io.imread(image_path)
    mask = io.imread(mask_path, as_gray=True).astype(np.uint8) * 255  # Ensure mask is binary

    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray')
    mask_overlay = ax.imshow(mask, cmap='jet', alpha=0.5)
    ax.set_title('Image with Mask Overlay')

    toggle_ax = plt.axes([0.7, 0.01, 0.1, 0.075])
    save_ax = plt.axes([0.81, 0.01, 0.1, 0.075])
    btn_toggle = Button(toggle_ax, 'Toggle')
    btn_save = Button(save_ax, 'Save')

    btn_toggle.on_clicked(toggle_value)
    btn_save.on_clicked(lambda event: save_mask(event, mask, mask_path))

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_release_event', onrelease)
    fig.canvas.mpl_connect('motion_notify_event', onmotion)

    plt.show()

# Loop over the images and find corresponding masks
for filename in os.listdir(image_dir):
    if filename.endswith('.tiff'):  # Adjust the extension according to your image files
        base_name = os.path.splitext(filename)[0]
        image_path = os.path.join(image_dir, filename)
        mask_filename = base_name + '_label.tiff'  # Adjust the suffix according to your naming convention
        mask_path = os.path.join(mask_dir, mask_filename)

        print(f"Processing {image_path} and {mask_path}")

        if os.path.exists(mask_path):
            edit_mask(image_path, mask_path)
        else:
            print(f"No corresponding mask found for {filename}")

print("All images have been processed.")