import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

def parse_datetime(datetime_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse a datetime string into a datetime object."""
    return datetime.strptime(datetime_str, format_str)

def calculate_time_differences_in_hours(datetimes, first_datetime):
    """Calculate the time differences in hours relative to the first datetime of each dataset."""
    return [(dt - first_datetime).total_seconds() / 3600.0 for dt in datetimes]

def filter_data_by_hour(time_differences, data, max_hour=25):
    """Filter data points to include only up to a specified hour."""
    return [(time, value) for time, value in zip(time_differences, data) if time <= max_hour]

def plot_cell_counts(output_filename, save_path=None, control_data=None, sheet_name=None):
    """Plot cell counts and control CFU from an Excel file against time with a logarithmic y-axis and intercept detection."""
    # Read the Excel file and specify the sheet
    df = pd.read_excel(output_filename, sheet_name=sheet_name)
    
    df_sorted = df.sort_values(by='Folder Name').dropna(subset=['Folder Name', 'Concentration(per ml)'])
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Concentration(per ml)'].tolist()

    # Parse and calculate time differences for the main data
    main_datetimes = [parse_datetime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_datetimes[0])

    # Filter main data to include only up to hour 50
    main_data_filtered = filter_data_by_hour(main_time_differences, cell_counts_from_excel, max_hour=50)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time (hours since start)')
    ax1.set_ylabel('Cell Concentration (CFU/mL)', color='black')
    ax1.set_yscale('log')  # Set y-axis to logarithmic scale
    ax1.tick_params(axis='y', labelcolor='black')

    # Plot device cell count data (Device Curve)
    device_times = [time for time, _ in main_data_filtered]
    device_counts = [count for _, count in main_data_filtered]
    ax1.plot(device_times, device_counts, marker='o', color='tab:blue', label='Device Cell Count')

    # Control data curve
    if control_data:
        control_datetimes = [parse_datetime(time_point, "%Y-%m-%d %H:%M:%S") for time_point, _ in control_data]
        
        # Use main data start time for consistency
        control_time_differences = calculate_time_differences_in_hours(control_datetimes, main_datetimes[0])
        control_data_filtered = filter_data_by_hour(control_time_differences, [cfu for _, cfu in control_data], max_hour=50)

        control_times = [time for time, _ in control_data_filtered]
        control_counts = [cfu for _, cfu in control_data_filtered]

        # Plot control CFU data on the same y-axis
        ax1.plot(control_times, control_counts, marker='o', color='tab:red', label='Control CFU', linestyle='--')

        # Interpolate both curves in log-space
        device_interp = interp1d(device_times, np.log(device_counts), kind='linear', bounds_error=False, fill_value="extrapolate")
        control_interp = interp1d(control_times, np.log(control_counts), kind='linear', bounds_error=False, fill_value="extrapolate")

        # Define a refined time range where the two curves might intersect
        min_time = max(min(device_times), min(control_times))
        max_time = min(max(device_times), max(control_times))

        # Define the function to find the intersection point
        def find_intersection(time):
            return device_interp(time) - control_interp(time)

        # Use a better initial guess based on visual inspection
        initial_guess = 16  # Adjusted initial guess slightly to the right of 15

        # Solve for the intersection point within the defined range
        intersection_time = fsolve(find_intersection, x0=initial_guess)[0]
        intersection_cfu_log = device_interp(intersection_time)
        intersection_cfu = np.exp(intersection_cfu_log)
        
        # Add the annotation at the intersection, move label to the left
        ax1.scatter(intersection_time, intersection_cfu, color='black', marker='x', s=100, label='Intersection Point')
        ax1.annotate(f'OD 0.8',
                     xy=(intersection_time, intersection_cfu),
                     xytext=(intersection_time - 5, intersection_cfu * 1.5),
                     arrowprops=dict(facecolor='black', arrowstyle="->"),
                     fontsize=12, color='black',
                     horizontalalignment='right')

    # Display legends
    ax1.legend(loc='upper left')

    plt.title('MG1655 Cell Concentration VS Time (CFU & OD)')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, format='tiff', dpi=300)
        print(f"Plot saved as TIFF at: {save_path}")

    plt.show()

# Example usage
output_filename = r'/home/urte/3D modeller/3d_cell_detector/data/raw/mg1655/01_03_cell_counts.xlsx'
save_path = 'ad38_counts.tiff'
sheet_name = 'Sheet1'  # Specify your sheet name here

# Control data in CFU values
control_data = [
    ('2024-03-01 15:43:00', 3.25*10**6),
    ('2024-03-01 18:03:00', 3.40*10**6),
    ('2024-03-01 23:26:00', 7.67*10**6),
    ('2024-03-02 05:10:00', 1.07*10**8),
    ('2024-03-02 06:21:00', 1.69*10**8),    
    ('2024-03-02 10:15:00', 1.88*10**9),
    ('2024-03-02 12:38:00', 1.98*10**9),
    ('2024-03-02 15:05:00', 5.44*10**9),
    ('2024-03-03 10:07:00', 6.42*10**10),
]

plot_cell_counts(output_filename, save_path, control_data, sheet_name)

'''
1:200 dilution

OD values for mg1655
15:43: 0.016
18:03 0.024
23:26 0.169
05:10 0.639
06:21 0.721
10:15 1.15
12:38 1.16
15:05 1.34
10:07 1.78

OD values for ad38
11:39:0.023
12:23 0.024
15:31 0.034
19:33 0.116
21:13 0.214
00:01 0.442
02:18 0.624
07:49 0.980
13:28 1.13
23:23 1.23

AD38
    ('2024-03-01 11:39:00', 1.83*10**7),
    ('2024-03-01 12:23:00', 1.84*10**7),
    ('2024-03-01 15:31:00', 1.90*10**7),
    ('2024-03-01 19:33:00', 2.51*10**7),
    ('2024-03-01 21:13:00', 3.50*10**7),    
    ('2024-03-02 00:01:00', 7.57*10**7),
    ('2024-03-02 02:18:00', 1.41*10**8),
    ('2024-03-02 07:49:00', 4.72*10**8),
    ('2024-03-02 13:28:00', 7.85*10**8),
    ('2024-03-02 23:23:00', 1.23*10**9)

MG1655
    ('2024-03-01 15:43:00', 3.25*10**6),
    ('2024-03-01 18:03:00', 3.40*10**6),
    ('2024-03-01 23:26:00', 7.67*10**6),
    ('2024-03-02 05:10:00', 1.07*10**8),
    ('2024-03-02 06:21:00', 1.69*10**8),    
    ('2024-03-02 10:15:00', 1.88*10**9),
    ('2024-03-02 12:38:00', 1.98*10**9),
    ('2024-03-02 15:05:00', 5.44*10**9),
    ('2024-03-03 10:07:00', 6.42*10**10),
]
'''
