import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_uv_vis(directory, output_dir=None, x_min=0, x_max=30):
    # Find all CSV files
    csv_files = glob.glob(f'{directory}/*.csv')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot each file
    for file in csv_files:
        # Read CSV, skipping first row
        df = pd.read_csv(file, skiprows=1)
        plt.plot(df['Time/sec'], df['Abs'], label=os.path.basename(file))
    
    plt.xlabel('Time (sec)')
    plt.ylabel('OD')
    plt.title('OD values with 2mg/ml starch VS time')
    plt.legend(loc='upper right')
    plt.grid(False)
    
    # Set y-axis limits
    plt.xlim(x_min, x_max)
    
    # Save or show
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/uv_vis_plot.png')
    else:
        plt.show()

# Example usage
plot_uv_vis('/home/urte/Spectrophotomer_starch/2mgml/UV1280/CData/')