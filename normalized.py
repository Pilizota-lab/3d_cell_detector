from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def parse_datetime(datetime_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse a datetime string into a datetime object."""
    return datetime.strptime(datetime_str, format_str)

def calculate_time_differences_in_hours(datetimes, first_datetime):
    """Calculate the time differences in hours relative to the first datetime of each dataset."""
    return [(dt - first_datetime).total_seconds() / 3600.0 for dt in datetimes]

def filter_data_by_hour(time_differences, data, max_hour=35):
    """Filter data points to include only up to a specified hour."""
    return [(time, value) for time, value in zip(time_differences, data) if time <= max_hour]

def calculate_growth_rates(time_differences, data):
    """Calculate growth rates as differences between successive data points over time intervals."""
    growth_rates = [(data[i+1] - data[i]) / (time_differences[i+1] - time_differences[i]) for i in range(len(data)-1)]
    return growth_rates

def normalize_to_max_growth_rate(time_differences, data):
    """Normalize the data relative to the maximum growth rate."""
    growth_rates = calculate_growth_rates(time_differences, data)
    max_growth_rate = max(growth_rates)
    normalized_data = [value / max_growth_rate for value in data]
    return normalized_data

def plot_cell_counts(output_filename, save_path=None, control_data=None, max_hour=None, font_size=14):
    """Plot cell counts and control OD from an Excel file against time independently, with legends."""
    df = pd.read_excel(output_filename)
    df_sorted = df.sort_values(by='Folder Name')
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Scaling Factor'].tolist()

    # Parse and calculate time differences for the main data
    main_datetimes = [parse_datetime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_datetimes[0])

    # Filter main data if max_hour is specified
    if max_hour is not None:
        main_data_filtered = filter_data_by_hour(main_time_differences, cell_counts_from_excel, max_hour)
        main_time_differences, cell_counts_from_excel = zip(*main_data_filtered)
    
    # Normalize main data to maximum growth rate
    normalized_main_counts = normalize_to_max_growth_rate(main_time_differences, cell_counts_from_excel)
    y_label = 'Cell Concentration (normalized to max growth rate)'

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time (hours since start)', fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.plot(main_time_differences, normalized_main_counts, marker='o', color='tab:blue', label='Cell Count')
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)

    # Plot control data if available
    if control_data:
        control_datetimes = [parse_datetime(time_point, "%Y-%m-%d %H:%M:%S") for time_point, _ in control_data]
        control_time_differences = calculate_time_differences_in_hours(control_datetimes, control_datetimes[0])

        # Filter control data if max_hour is specified
        if max_hour is not None:
            control_data_filtered = filter_data_by_hour(control_time_differences, [od for _, od in control_data], max_hour)
            control_time_differences, control_ODs = zip(*control_data_filtered)
        else:
            control_ODs = [od for _, od in control_data]

        # Normalize control data to maximum growth rate
        normalized_control_od = normalize_to_max_growth_rate(control_time_differences, control_ODs)
        control_y_label = 'Control OD (normalized to max growth rate)'

        ax2 = ax1.twinx()
        ax2.set_ylabel(control_y_label, color='tab:red', fontsize=14)
        ax2.plot(control_time_differences, normalized_control_od, marker='o', color='tab:red', label='Control OD')
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=font_size)

    # Display legends for both datasets
    lines, labels = ax1.get_legend_handles_labels()
    if control_data:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper left', fontsize=font_size)

    plt.title('MG1655 Cell Concentration + 760mM Sucrose VS Time', fontsize=font_size)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, format='tiff', dpi=300)
        print(f"Plot saved as TIFF at: {save_path}")

    plt.show()

# Example usage
output_filename = r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_02_24_model_temporal_4_SUCROSE.xlsx'
save_path = 'ad38_counts.tiff'
control_data = [
    ('2024-03-01 15:43:00', 1.28*10**7),
    ('2024-03-01 18:03:00', 1.92*10**7),
    ('2024-03-01 23:26:00', 1.35*10**8),
    ('2024-03-02 05:10:00', 5.11*10**8),
    ('2024-03-02 06:21:00', 5.77*10**8),
    ('2024-03-02 10:15:00', 9.20*10**8),
    ('2024-03-02 12:38:00', 9.28*10**8),
    ('2024-03-02 15:05:00', 1.07*10**9),
    ('2024-03-03 10:07:00', 1.42*10**9 ),
]

# You can specify the maximum hour, font size, and normalization method
plot_cell_counts(output_filename, save_path, max_hour=None, font_size=16)
