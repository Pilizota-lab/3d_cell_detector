import support_functions as sf
import os
from tqdm import tqdm  # Import tqdm for progress bar

base_directory = r'Z:\Urte\PhD\Imaging Device\Captures\Cells\BW25113+pWR20\14_10_23'
model = sf.load_working_model('models/Urte_4')

all_directories = [dir_name for dir_name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, dir_name))]
sorted_directories = sorted(all_directories, key=lambda x: (x[:8], x[9:]))

folder_names = []
cell_counts = []
print(sorted_directories)
'''
for directory in tqdm(sorted_directories, desc="Processing directories"):
    folder_path = os.path.join(base_directory, directory)
    images = sf.load_images(folder_path)

    #Diameter of cells can be adjsuted (30 pixels is the automatic input for cyto model; cellpose allows to estimate the diameter)
    channels = [[0,0]]
    masks, flows, styles = model.eval(images, diameter=37, channels=channels, compute_masks=True)

    # Ensure consistent directory paths using os.path.join
    results_directory = os.path.join(base_directory, directory, 'results')
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    # Save masks and overlays
    sf.save_masks_overlay(images, masks, results_directory)

    # Save 3D stack
    sf.create_3d_stack(masks, results_directory)

    # Count objects and save visualizations
    number_of_cells = sf.count_objects(masks, results_directory)
    
    folder_names.append(directory)
    cell_counts.append(number_of_cells)
    '''

sf.save_to_excel(folder_names, cell_counts)
