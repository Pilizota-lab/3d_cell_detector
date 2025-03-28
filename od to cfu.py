import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def convert_od_to_cfu(od_values):
    """
    Convert optical density (OD) values to CFU/mL.
    
    Parameters:
    od_values (list or float): OD value(s)
    
    Returns:
    list or float: Corresponding CFU/mL value(s)
    """
    return 10 ** ((np.array(od_values) - (-4.9032)) / 0.6783)

def process_od_data(input_file, output_file=None):
    """
    Process OD data from a CSV file.
    
    Parameters:
    input_file (str): Path to input CSV file with OD values
    output_file (str, optional): Path to save results
    
    Returns:
    pandas.DataFrame: Processed data with CFU/mL values
    """
    # Read input file
    df = pd.read_csv(input_file)
    
    # Assume the OD column is named 'OD' - adjust if different
    df['CFU/mL'] = convert_od_to_cfu(df['OD'])
    
    # Optional: save to output file
    if output_file:
        df.to_csv(output_file, index=False)
    
    return df

def plot_od_vs_cfu(df):
    """
    Create a scatter plot of OD vs CFU/mL.
    
    Parameters:
    df (pandas.DataFrame): Dataframe with OD and CFU/mL columns
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df['OD'], df['CFU/mL'], alpha=0.7)
    plt.xlabel('Optical Density (OD)')
    plt.ylabel('Cell Concentration (CFU/mL)')
    plt.title('OD vs Cell Concentration')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    od_values = [0.250, 0.659, 0.862, 1.032, 1.128, 1.316, 1.123]
    
    # Convert and print individual OD values
    print("Individual OD to CFU/mL Conversion:")
    for od in od_values:
        cfu = convert_od_to_cfu(od)
        print(f"OD: {od:.3f} → {cfu:.2e} CFU/mL")
    
    # Optional: Process from CSV and visualize (uncomment if needed)
    # df = process_od_data('od_input.csv', 'od_output.csv')
    # plot_od_vs_cfu(df)

if __name__ == "__main__":
    main()