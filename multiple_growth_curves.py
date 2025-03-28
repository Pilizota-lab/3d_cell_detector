import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

def parse_datetime_from_folder(folder_name, format_str="%y_%m_%d_%H_%M_%S"):
    """Parse a datetime from folder name format."""
    return datetime.strptime(folder_name, format_str)

def process_excel_file(excel_path, sheet_name='Sheet1'):
    """
    Process a single Excel file and return time differences and cell counts.
    
    Parameters:
    excel_path -- Path to the Excel file
    sheet_name -- Name of the sheet containing the data
    
    Returns:
    list of (time_difference, cell_count) pairs
    """
    # Read the Excel file
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    
    # Ensure there are no missing values in required columns
    df_sorted = df.sort_values(by='Folder Name').dropna(subset=['Folder Name', 'Concentration(ml)'])
    
    # Extract and parse datetimes from folder names
    datetimes = [parse_datetime_from_folder(name) for name in df_sorted['Folder Name']]
    cell_counts = df_sorted['Concentration(ml)'].tolist()
    
    # Calculate time differences from the first measurement
    start_time = datetimes[0]
    time_differences = [(dt - start_time).total_seconds() / 3600.0 for dt in datetimes]
    
    return list(zip(time_differences, cell_counts))

def plot_multiple_growth_curves(excel_files, labels=None, title='Growth Curves', 
                              save_path=None, max_hour=50, sheet_name='Sheet1'):
    """
    Plot multiple growth curves from Excel files on the same graph.
    
    Parameters:
    excel_files -- List of paths to Excel files
    labels -- List of labels for each dataset (if None, will use file names)
    title -- Title for the plot
    save_path -- Optional path to save the plot
    max_hour -- Maximum time in hours to show on the plot
    sheet_name -- Name of the sheet containing the data in each Excel file
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate labels from filenames if not provided
    if labels is None:
        labels = [os.path.basename(file).split('.')[0] for file in excel_files]
    
    # Color map for multiple curves
    colors = plt.cm.tab10(np.linspace(0, 1, len(excel_files)))
    
    # Process and plot each Excel file
    for excel_file, label, color in zip(excel_files, labels, colors):
        try:
            # Process the Excel file
            data = process_excel_file(excel_file, sheet_name)
            
            # Filter data points up to max_hour
            filtered_data = [(time, value) for time, value in data if time <= max_hour]
            
            if filtered_data:
                times, values = zip(*filtered_data)
                ax.plot(times, values, marker='o', color=color, label=label)
            else:
                print(f"No data points to plot for {label} after filtering to {max_hour} hours.")
                
        except Exception as e:
            print(f"Error processing {excel_file}: {str(e)}")
    
    # Set up the plot
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cell Concentration (CFU/ml)')
    ax.set_yscale('log')
    ax.set_ylim(1e7, 1e9)
    ax.legend(loc='upper left')
    ax.grid(True, which='both', ls='-', alpha=0.2)
    plt.title(title)
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='tiff', dpi=300)
        print(f"Plot saved as TIFF at: {save_path}")
    
    plt.show()

# Example usage:
excel_files = [
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_02_24_model_temporal_4_SUCROSE.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/cell_counts_summary_temp_5_ad38.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/23_12_23_model_Urte_5.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/28_02_24_model_temporal_4.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_03_cell_counts.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/bw25113/cell_counts_14_10_model_Urte_5.xlsx'
]


# Optional: provide custom labels for each dataset
labels = [
    '01_02_24 (MG1655)',
    '01_02_24 (AD38)',
    #'23_12_23 (colony)',
    #'28_02_24 (colony)',
    #'28_01_25 (start OD 1.02; frozen stock)',
    #'14_10_23(frozen stock)'
    # Add more labels as needed
]


# Plot the data
plot_multiple_growth_curves(
    excel_files=excel_files,
    labels=labels,
    title='MG1655 Growth Curves',
    save_path='ad38_multiple_growth_curves.tiff',
    max_hour=150,
    sheet_name='Sheet1'
)

'''
# Example usage:
excel_files = [
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/01_12_23/24_02_28_15_40_47.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/08_12_23/08_12_23_model_Urte_5.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/11_18_24/cell_counts_summary.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/24_11_24/cell_counts_summary.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/28_01_25/cell_counts_summary_temp_5.xlsx'
]

# Optional: provide custom labels for each dataset
labels = [
    #'01_12_23 ',
    #'08_12_23',
    '11_18_24 (start OD 3.3; colony)',
    '24_11_24 (start OD 1.76; colony)',
    '28_01_25 (start OD 1.02; frozen stock)'
    # Add more labels as needed
]

# Example usage:
excel_files = [
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/01_12_23/24_02_28_15_40_47.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/08_12_23/08_12_23_model_Urte_5.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/2mg_mlstarch/26_11_24/cell_counts_summary_temp_5.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/2mg_mlstarch/30_01_25/cell_counts_summary_temp_5.xlsx'
]

# Optional: provide custom labels for each dataset
labels = [
    #'01_12_23 ',
    #'08_12_23',
    '26_11_24 (start OD 1.1; colony)',
    '30_01_24 (start OD 0.95, frozen stock)'
    # Add more labels as needed
]
# Example usage:
excel_files = [
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_03_cell_counts.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_02_24_model_temporal_4.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/23_12_23_model_Urte_5.xlsx',
    r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/28_02_24_model_temporal_4.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_03_cell_counts.xlsx',
    #r'/home/urte/3D modeller/3d_cell_detector/data/raw/bw25113/cell_counts_14_10_model_Urte_5.xlsx'
]


# Optional: provide custom labels for each dataset
labels = [
    '01_03_23 (colony)',
    '01_02_24 (colony)',
    '23_12_23 (colony)',
    '28_02_24 (colony)',
    #'28_01_25 (start OD 1.02; frozen stock)',
    #'14_10_23(frozen stock)'
    # Add more labels as needed
]
'''