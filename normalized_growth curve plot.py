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

def normalize_to_max_value(data):
    """Normalize the data relative to the maximum value in the dataset."""
    max_value = max(data)
    normalized_data = [value / max_value for value in data]
    return normalized_data

def plot_cell_counts(output_filename, save_path=None, control_data=None, max_hour=None, font_size=14):
    """Plot cell counts and control OD from an Excel file against time, normalized to their maximum values."""
    df = pd.read_excel(output_filename)
    df_sorted = df.sort_values(by='Folder Name')
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Scaling Factor'].tolist()

    main_datetimes = [parse_datetime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_datetimes[0])

    if max_hour is not None:
        main_data_filtered = filter_data_by_hour(main_time_differences, cell_counts_from_excel, max_hour)
        if main_data_filtered:
            main_time_differences, cell_counts_from_excel = zip(*main_data_filtered)
        else:
            print(f"No data within {max_hour} hours. Check the time differences or `max_hour` value.")
            return  # Exit if no data is available within the max_hour range

    normalized_main_counts = normalize_to_max_value(cell_counts_from_excel)
    y_label = 'Cell Concentration (normalized to max cell number)'

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time (hours since start)', fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.plot(main_time_differences, normalized_main_counts, marker='o', color='tab:blue', label='Cell Count')
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)

    if control_data:
        control_datetimes = [parse_datetime(time_point, "%Y-%m-%d %H:%M:%S") for time_point, _ in control_data]
        control_time_differences = calculate_time_differences_in_hours(control_datetimes, control_datetimes[0])

        if max_hour is not None:
            control_data_filtered = filter_data_by_hour(control_time_differences, [od for _, od in control_data], max_hour)
            if control_data_filtered:
                control_time_differences, control_ODs = zip(*control_data_filtered)
            else:
                print(f"No control data within {max_hour} hours. Check the control time differences or `max_hour` value.")
                return  # Exit if no control data is available within the max_hour range

        normalized_control_od = normalize_to_max_value(control_ODs)
        control_y_label = 'Control OD (normalized to max cell number)'

        ax2 = ax1.twinx()
        ax2.set_ylabel(control_y_label, color='tab:red', fontsize=14)
        ax2.plot(control_time_differences, normalized_control_od, marker='o', color='tab:red', label='Control OD')
        ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=font_size)

    lines, labels = ax1.get_legend_handles_labels()
    if control_data:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    ax1.legend(lines, labels, loc='upper left', fontsize=font_size)

    plt.title('AD38 (delta MotAB MG1655) Cell Concentration VS Time', fontsize=font_size)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, format='tiff', dpi=300)
        print(f"Plot saved as TIFF at: {save_path}")

    plt.show()

# Example usage
output_filename = r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/01_12_23/24_02_28_15_40_47.xlsx'
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
plot_cell_counts(output_filename, save_path, control_data, max_hour=31, font_size=16)
