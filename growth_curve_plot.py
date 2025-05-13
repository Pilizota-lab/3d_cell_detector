import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def parse_datetime(datetime_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse a datetime string into a datetime object."""
    return datetime.strptime(datetime_str, format_str)

def calculate_time_differences_in_hours(datetimes, reference_datetime):
    """Calculate the time differences in hours relative to the given reference datetime."""
    return [(dt - reference_datetime).total_seconds() / 3600.0 for dt in datetimes]

def filter_data_by_hour(time_differences, data, max_hour=50):
    """Filter data points to include only up to a specified hour."""
    return [(time, value) for time, value in zip(time_differences, data) if time <= max_hour]

def process_control_data_separate_timeline(control_data, max_hour=50):
    """
    Process a control dataset (undiluted or diluted) with its own timeline.
    Returns filtered (time, value) pairs or an empty list if no data.
    """
    if not control_data:
        return []

    control_datetimes = [parse_datetime(time_point, "%Y-%m-%d %H:%M:%S") for time_point, _ in control_data]
    control_values = [od for _, od in control_data]
    control_start = control_datetimes[0]
    control_time_differences = calculate_time_differences_in_hours(control_datetimes, control_start)
    control_data_filtered = filter_data_by_hour(control_time_differences, control_values, max_hour=max_hour)
    return control_data_filtered

def plot_cell_counts_separate_timelines(output_filename, 
                                        save_path=None, 
                                        sheet_name=None, 
                                        control_data_diluted=None, 
                                        control_data_undiluted=None,
                                        std_devs=None,
                                        max_hour=50):
    """
    Plot the main dataset along with undiluted and diluted control datasets, 
    each having their own timeline starting at zero.

    Each dataset's time is calculated independently from its first data point.
    """
    # Read the Excel file and specify the sheet
    df = pd.read_excel(output_filename, sheet_name=sheet_name)
    
    # Ensure there are no missing values in 'Folder Name' and 'Concentration(ml)'
    df_sorted = df.sort_values(by='Folder Name').dropna(subset=['Folder Name', 'Concentration(ml)'])
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Concentration(ml)'].tolist()

    # Parse the main dataset datetimes (assuming format %y_%m_%d_%H_%M_%S)
    main_datetimes = [datetime.strptime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]

    # Calculate main timeline (starting at its own first measurement)
    main_start = main_datetimes[0]
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_start)
    main_data_filtered = filter_data_by_hour(main_time_differences, cell_counts_from_excel, max_hour=max_hour)

    # Process undiluted control data, if provided
    undiluted_data_filtered = process_control_data_separate_timeline(control_data_undiluted, max_hour=max_hour)

    # Process diluted control data, if provided
    diluted_data_filtered = process_control_data_separate_timeline(control_data_diluted, max_hour=max_hour)

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Time (hours since each dataset\'s start)')
    ax1.set_ylabel('Cell Concentration (CFU/ml)', color='tab:blue')

    # Plot the main dataset
    if main_data_filtered:
        ax1.plot(
            [time for time, _ in main_data_filtered],
            [count for _, count in main_data_filtered],
            marker='o',
            color='tab:blue',
            label='Cell Count (Device)'
        )
    else:
        print("No main data points to plot after filtering.")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_yscale('log')
    ax1.set_ylim(1e6, 1e9)  # Adjusted y-axis limits for log scale

    # For controls, we'll use a secondary axis
    if undiluted_data_filtered or diluted_data_filtered:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Control Cell Concentration (CFU/ml)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.set_yscale('log')
        ax2.set_ylim(1e6, 1e9)

        # Plot diluted control data with error bars
        if diluted_data_filtered and std_devs:
            # Ensure std_devs matches the number of data points
            if len(std_devs) == len(diluted_data_filtered):
                ax2.errorbar(
                    [time for time, _ in diluted_data_filtered],
                    [OD for _, OD in diluted_data_filtered],
                    #yerr=std_devs,
                    fmt='^', 
                    color='tab:red',
                    linestyle='-',
                    capsize=5,
                    label='CFU Count (Control)'
                )
            else:
                print("Warning: Number of std_devs does not match number of diluted control data points.")
                # Fallback to plotting without error bars
                ax2.plot(
                    [time for time, _ in diluted_data_filtered],
                    [OD for _, OD in diluted_data_filtered],
                    marker='o',
                    color='tab:red',
                    label='Diluted Control'
                )
        elif diluted_data_filtered:
            ax2.plot(
                [time for time, _ in diluted_data_filtered],
                [OD for _, OD in diluted_data_filtered],
                marker='o',
                color='tab:red',
                label='Diluted Control'
            )

        # Plot undiluted control data
        if undiluted_data_filtered:
            ax2.plot(
                [time for time, _ in undiluted_data_filtered],
                [OD for _, OD in undiluted_data_filtered],
                marker='s',
                color='tab:red',
                label='Undiluted Control'
            )
        else:
            print("No undiluted control data points after filtering.")

    # Manage legends
    lines, labels = ax1.get_legend_handles_labels()
    if undiluted_data_filtered or diluted_data_filtered:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.legend(lines, labels, loc='upper left')

    plt.title('AD38 (deltaMotAB MG1655) Cell Concentration +5 mg/ml Starch VS Time')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, format='tiff', dpi=300)
        print(f"Plot saved as TIFF at: {save_path}")

    plt.show()

# Example usage (replace with your actual file paths and data):
output_filename = r'/home/urte/3D modeller/3d_cell_detector/data/raw/ad38/5mg_mlstarch/cell_counts_summary_temp_5.xlsx'
save_path = 'ad38_counts_separate_timeline_controls.tiff'
sheet_name = 'Sheet1'
#std_devs = [3.77E+07,5.89E+07,3.35E+08,7.36E+08,1.25E+09,1.42E+09,7.72E+08,2.27E+09]

control_data_diluted = [
    ('2024-11-18 17:10:00', 7.50*10**7),
    ('2024-11-19 10:50:00', 1.98*10**8),
    ('2024-11-19 13:50:00', 2.59*10**8),
    ('2024-11-19 16:50:00', 3.10*10**8),
    ('2024-11-20 16:50:00', 3.38*10**8),
    ('2024-11-20 17:50:00', 3.95*10**8),
    ('2024-11-21 18:54:00', 3.37*10**8)
]

control_data_undiluted=[
    ('2024-03-01 17:29:00', 1.75*10**8),
    ('2024-03-02 10:02:00', 2.86*10**8),
    ('2024-03-02 13:40:00', 6.70*10**8),
    ('2024-03-02 17:15:00', 1.72*10**9),
    ('2024-03-02 18:15:00', 2.11*10**9),    
    ('2024-03-03 12:57:00', 5.98*10**9),
    ('2024-03-03 16:10:00', 1.27*10**11),
    ('2024-03-03 18:20:00', 1.27*10**11)


]

plot_cell_counts_separate_timelines(
    output_filename,
    save_path=save_path,
    sheet_name=sheet_name,
    control_data_diluted=control_data_diluted,
    control_data_undiluted=control_data_undiluted,
    #std_devs=std_devs,
    max_hour=50
)


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

AD28 01/18/24 diluted
    ('2024-03-01 18:40:00', 1.40*10**8),
    ('2024-03-02 10:07:00', 6.60*10**8),
    ('2024-03-02 13:07:00', 1.10*10**10),
    ('2024-03-02 16:51:00', 2.20*10**9),
    ('2024-03-02 18:17:00', 2.16*10**10),    
    ('2024-03-03 13:40:00', 1.37*10**11),
    ('2024-03-03 16:05:00', 1.58*10**10),
    ('2024-03-03 17:37:00', 2.88*10**10)


AD28 18/11/24 diluted
    ('2024-03-01 17:29:00', 2.70*10**6),
    ('2024-03-02 10:02:00', 4.62*10**7),
    ('2024-03-02 13:40:00', 1.22*10**8),
    ('2024-03-02 17:15:00', 2.05*10**8),
    ('2024-03-02 18:15:00', 2.23*10**8),    
    ('2024-03-03 12:57:00', 3.15*10**8),
    ('2024-03-03 16:10:00', 5.85*10**8),
    ('2024-03-03 18:20:00', 5.58*10**8)

AD28 18/11/24 undiluted
    ('2024-03-01 17:29:00', 1.75*10**7),
    ('2024-03-02 10:02:00', 2.86*10**7),
    ('2024-03-02 13:40:00', 6.70*10**7),
    ('2024-03-02 17:15:00', 1.72*10**8),
    ('2024-03-02 18:15:00', 2.11*10**8),    
    ('2024-03-03 12:57:00', 5.98*10**8),
    ('2024-03-03 16:10:00', 1.27*10**10),
    ('2024-03-03 18:20:00', 1.27*10**10)

AD38 24/11/24 diluted
    
    ('2024-11-18 18:22:00', 1.74*10**7),
    ('2024-11-19 11:20:00', 1.04*10**8),
    ('2024-11-19 13:56:00', 2.24*10**8),
    ('2024-11-19 15:45:00', 2.94*10**8),
    ('2024-11-19 16:50:00', 3.33*10**8),
    ('2024-11-19 17:50:00', 3.70*10**8),
    ('2024-11-19 18:54:00', 3.99*10**8),
    ('2024-11-20 12:43:00', 8.99*10**8),
    ('2024-11-20 16:09:00', 9.62*10**8),
    ('2024-11-20 18:10:00', 9.62*10**8)

 AD38 24/11/24 undiluted   
    ('2024-11-18 18:22:00', 1.74*10**7),
    ('2024-11-19 11:20:00', 1.04*10**8),
    ('2024-11-19 13:56:00', 2.24*10**8),
    ('2024-11-19 15:45:00', 2.94*10**8),
    ('2024-11-19 16:50:00', 3.33*10**8),
    ('2024-11-19 17:50:00', 3.70*10**8),
    ('2024-11-19 18:54:00', 3.99*10**8),
    ('2024-11-20 12:42:00', 8.77*10**8),
    ('2024-11-20 16:08:00', 1.04*10**9),
    ('2024-11-20 18:10:00', 1.04*10**9)

AD38 26/11/24 2mg/ml starch  
    ('2024-11-18 18:22:00', 1.74*10**7),
    ('2024-11-19 11:20:00', 1.04*10**8),
    ('2024-11-19 13:56:00', 2.24*10**8),
    ('2024-11-19 15:45:00', 2.94*10**8),
    ('2024-11-19 16:50:00', 3.33*10**8),
    ('2024-11-19 17:50:00', 3.70*10**8),
    ('2024-11-19 18:54:00', 3.99*10**8),
    ('2024-11-20 12:42:00', 8.77*10**8),
    ('2024-11-20 16:08:00', 1.04*10**9),
    ('2024-11-20 18:10:00', 1.04*10**9)

AD38 09/12/24 5mg/ml starch  
    ('2024-11-18 17:10:00', 3.96*10**7),
    ('2024-11-19 10:50:00', 1.59*10**8),
    ('2024-11-19 13:50:00', 3.16*10**8),
    ('2024-11-19 16:50:00', 5.62*10**8),
    ('2024-11-20 16:50:00', 7.79*10**8),
    ('2024-11-20 17:50:00', 1.48*10**9),
    ('2024-11-21 18:54:00', 7.66*10**8)


AD38 30_01_25 2mg/ml starch  
    ('2024-11-18 19:32:00', 3.96*10**7),
    ('2024-11-19 10:13:00', 1.59*10**8),
    ('2024-11-19 12:40:00', 3.16*10**8),
    ('2024-11-19 16:43:00', 5.62*10**8),
    ('2024-11-19 18:00:00', 7.79*10**8),
    ('2024-11-20 15:00:00', 1.48*10**9),
    ('2024-11-20 18:00:00', 7.66*10**8)

 AD38 25/01/25 diluted   
    ('2024-03-01 18:40:00', 2.10*10**6),
    ('2024-03-02 10:07:00', 1.23*10**8),
    ('2024-03-02 13:07:00', 1.89*10**8),
    ('2024-03-02 16:51:00', 2.70*10**8),
    ('2024-03-02 18:17:00', 3.33*10**8),    
    ('2024-03-03 13:40:00', 5.67*10**8),
    ('2024-03-03 16:05:00', 3.93*10**8),
    ('2024-03-03 17:37:00', 3.66*10**8)
'''
