import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Manually inputting the data from the image
data = {
    'OD': [0.46, 0.153, 0.018, 0.014, 0.007, 0.014, 0.015, 0.012, 0.001, 0.001],
    'Cell Number': [np.nan, np.nan, np.nan, np.nan, np.nan, 4.93e8, 6.6e8, 7.0e8, np.nan, np.nan]
}

df = pd.DataFrame(data)

# Plotting the data points
plt.figure(figsize=(10, 6))

# Plot points with known cell numbers
known_points = df.dropna()
plt.scatter(known_points['OD'], known_points['Cell Number'], color='blue', label='Known data points')

# Highlighting points without cell numbers
for index, row in df.iterrows():
    if pd.isna(row['Cell Number']):
        plt.scatter(row['OD'], 0, color='red', edgecolor='black', s=100, label='Need projection' if index == 0 else "")

plt.xlabel('OD')
plt.ylabel('Cell Number')
plt.title('OD vs Cell Number')
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)  # Ensuring the y-axis starts from 0
plt.show()
