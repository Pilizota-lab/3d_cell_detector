import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updated conversion function using constrained linear fit through origin
def convert_od_to_cfu(od_values):
    """
    Convert optical density (OD) values to CFU/mL using a linear fit through the origin.
    
    Parameters:
    od_values (list or float): OD value(s)
    
    Returns:
    list or float: Corresponding CFU/mL value(s)
    """
    slope = 3.00e8  # From regression through origin
    return np.array(od_values) * slope

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
    plt.scatter(df['OD'], df['CFU/mL'], alpha=0.7, label="Data Points")
    
    # Add fitted line from origin
    x_vals = np.linspace(0, df['OD'].max()*1.1, 100)
    y_vals = convert_od_to_cfu(x_vals)
    plt.plot(x_vals, y_vals, color='black', linestyle='--', label="Fit: CFU = 3.00e8 × OD")

    plt.xlabel('Optical Density (OD)')
    plt.ylabel('Cell Concentration (CFU/mL)')
    plt.title('OD vs Cell Concentration (Linear Fit Through Origin)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Example usage
    od_values = [0.250, 0.659, 0.862, 1.032, 1.128, 1.316, 1.123]
    
    print("Individual OD to CFU/mL Conversion:")
    for od in od_values:
        cfu = convert_od_to_cfu(od)
        print(f"OD: {od:.3f} → {cfu:.2e} CFU/mL")
        print (f'{cfu:.2e}' )
    
    # Optional: Process CSV and visualize
    # df = process_od_data('od_input.csv', 'od_output.csv')
    # plot_od_vs_cfu(df)

if __name__ == "__main__":
    main()

    '''
    AD38 start OD 3.3 (nov 24)
    od_values = [0.009, 0.154, 0.405, 0.683, 0.743, 1.05, 1.95, 1.95]

    ad38 start od 1.76 (nov24)
    od_values = [0.008, 0.536, 0.761, 0.841, 0.878, 0.909,0.931, 1.17, 1.19, 1.19]

    ad38 start od 1.0 (jan 2025)
    od_values = [0.007, 0.41, 0.63, 0.90, 1.11, 1.89, 1.31, 1.22]
    
    ad38 +2mg/ml starch od start 1 (jan 25)
    od_values = [0.329, 0.70, 1.00, 1.12, 1.32, 1.65, 1.57]

    ad38 +5mg/ml starch od start 1 (dec 24)
    od_values = [0.250, 0.659, 0.862, 1.032, 1.128, 1.316, 1.123]
    '''