import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def parse_datetime(datetime_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse a datetime string into a datetime object."""
    return datetime.strptime(datetime_str, format_str)

def calculate_time_differences_in_hours(datetimes, first_datetime):
    """Calculate the time differences in hours relative to the first datetime of each dataset."""
    return [(dt - first_datetime).total_seconds() / 3600.0 for dt in datetimes]

def filter_data_by_hour(time_differences, data, max_hour=50):
    """Filter data points to include only up to a specified hour."""
    return [(time, value) for time, value in zip(time_differences, data) if time <= max_hour]

def plot_cell_counts(output_filename, save_path=None, control_data=None):
    """Plot cell counts and control data from an Excel file against time, 
    with both y-axes sharing the same scale for cell concentration per ml."""
    df = pd.read_excel(output_filename)
    df_sorted = df.sort_values(by='Folder Name')
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Concentration( in volume of 0.0491cm^3)'].tolist()

    # Parse and calculate time differences for the main data
    main_datetimes = [parse_datetime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_datetimes[0])

    # Filter main data to include only up to hour 50
    main_data_filtered = filter_data_by_hour(main_time_differences, cell_counts_from_excel)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time (hours since start)')
    ax1.set_ylabel('Cell Concentration (number of cells/ml)', color='tab:blue')
    ax1.plot([time for time, _ in main_data_filtered], [count for _, count in main_data_filtered], marker='o', color='tab:blue', label='Cell Count')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()  # Create a second y-axis to share the same x-axis

    # Set explicit limits for the y-axes to 1.8*10^9
    y_max = 1.8 * 10**9  # Max limit for y-axis
    ax1.set_ylim([0, y_max])
    ax2.set_ylim([0, y_max])

    ax2.set_ylabel('Cell Concentration (number of cells/ml)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    if control_data:
        control_datetimes = [parse_datetime(time_point, "%Y-%m-%d %H:%M:%S") for time_point, _ in control_data]
        control_time_differences = calculate_time_differences_in_hours(control_datetimes, control_datetimes[0])
        control_data_filtered = filter_data_by_hour(control_time_differences, [od for _, od in control_data])

        # Plot control data on the second y-axis (ax2)
        ax2.plot([time for time, _ in control_data_filtered], [OD for _, OD in control_data_filtered], marker='o', color='tab:red', label='Control Data')
        
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        if control_data:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        else:
            ax1.legend(lines, labels, loc='upper left')

        plt.title('Cell Count + Control Data VS Time')
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, format='tiff', dpi=300)
            print(f"Plot saved as TIFF at: {save_path}")

        plt.show()

 # Example usage
output_filename = r'Y:\groups\pilizota\Urte\PhD\Imaging Device\Captures\Cells\mg1655\01_03_24\01_03_cell_counts.xlsx'  
save_path = '01_12_23_cell_counts.tiff'  # Corrected to '.tiff'

# Example control data (update with your actual data)
control_data = [
    ('2024-03-01 15:43:00', 1.28*10**7),
    ('2024-03-01 18:03:00', 1.92*10**7),
    ('2024-03-01 23:26:00', 1.35*10**8),
    ('2024-03-02 05:10:00', 5.11*10**8),
    ('2024-03-02 06:21:00', 5.77*10**8),
    ('2024-03-02 10:15:00', 9.20*10**8),
    ('2024-03-02 12:38:00', 9.28*10**8),
    ('2024-03-02 15:05:00', 1.07*10**9),
    ('2024-03-03 10:07:00', 1.42*10**9),
]


plot_cell_counts(output_filename, save_path, control_data)   