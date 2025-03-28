from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

def parse_datetime(datetime_str, format_str="%Y-%m-%d %H:%M:%S"):
    """Parse a datetime string into a datetime object."""
    return datetime.strptime(datetime_str, format_str)

def calculate_time_differences_in_hours(datetimes, first_datetime):
    """Calculate the time differences in hours relative to the first datetime."""
    return [(dt - first_datetime).total_seconds() / 3600.0 for dt in datetimes]

def plot_cell_counts(output_filename, control_data=None, control_time_format=None):
    """Plot cell counts and optionally control OD from an Excel file against time."""
    df = pd.read_excel(output_filename)
    df_sorted = df.sort_values(by='Folder Name')
    sorted_directories_from_excel = df_sorted['Folder Name'].tolist()
    cell_counts_from_excel = df_sorted['Cell Count'].tolist()

    # Parse the datetimes for the main data
    main_datetimes = [parse_datetime(dt, "%y_%m_%d_%H_%M_%S") for dt in sorted_directories_from_excel]

    # Calculate time differences relative to the first datetime in the main dataset
    main_time_differences = calculate_time_differences_in_hours(main_datetimes, main_datetimes[0])

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting Cell Count Data on primary y-axis
    ax1.set_xlabel('Time (hours since start)')
    ax1.set_ylabel('Cell Count', color='tab:blue')
    ax1.plot(main_time_differences, cell_counts_from_excel, marker='o', color='tab:blue', label='Imaging Device')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Plot control data if provided
    if control_data is not None and control_time_format is not None:
        # Parse the datetimes for control data
        control_datetimes = [parse_datetime(time_point, control_time_format) for time_point, _ in control_data]

        # Calculate time differences for control data
        control_time_differences = calculate_time_differences_in_hours(control_datetimes, control_datetimes[0])

        # Creating secondary y-axis for OD data
        ax2 = ax1.twinx()
        ax2.set_ylabel('OD', color='tab:red')
        ax2.plot(control_time_differences, [OD for _, OD in control_data], marker='o', color='tab:red', label='Control Flask')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Adding legend for both datasets
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    else:
        # Adding legend for cell count data only
        ax1.legend(loc='upper left')

    plt.title('MG1655 Cell Count + 760mM sucrose VS Time (M63 media; model temporal_4)')
    fig.tight_layout()
    # Uncomment and ensure save_path is defined before using
    # plt.savefig(save_path, format='tiff')
    plt.show()

# Example usage:
output_filename = r'X:\phD\captures\mg1655\06_02_24\06_02_24_model_temporal_4_counts.xlsx'
control_data = [('2023-12-23 12:13:00', 0.064), ...]  # Fill in your control data
control_time_format = "%Y-%m-%d %H:%M:%S"

plot_cell_counts(output_filename)
