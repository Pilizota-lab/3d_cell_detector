import os
import pandas as pd

def collect_and_save_counts(base_directory, model_identifier, output_file):
    folder_names = []
    cell_counts = []
    
    for dir_name in os.listdir(base_directory):
        dir_path = os.path.join(base_directory, dir_name)
        results_dir_name = f'results_model{model_identifier}'
        results_dir_path = os.path.join(dir_path, results_dir_name)
        count_file_path = os.path.join(results_dir_path, 'cell_count.txt')
        
        if os.path.isdir(dir_path) and os.path.exists(count_file_path):
            with open(count_file_path, 'r') as count_file:
                count = count_file.read().strip()
                folder_names.append(dir_name)
                cell_counts.append(int(count))
                
    # Save the results to an Excel file
    if folder_names and cell_counts:
        results_df = pd.DataFrame({'Folder Name': folder_names, 'Cell Count': cell_counts})
        results_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    else:
        print("No results to save.")

# Set parameters
base_directory = r'X:\phD\captures\mg1655\06_02_24'
model_identifier = 'temporal_4'
output_file = os.path.join(base_directory, '06_02_24_model_temporal_4_counts.xlsx')

# Execute function
collect_and_save_counts(base_directory, model_identifier, output_file)
