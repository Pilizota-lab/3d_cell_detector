import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Data for frozen stock
od_values_frozen = [4.24, 2.09, 1.18, 0.62, 0.38, 0.22, 0.10]
cfu_values_frozen = [1.14e10, 1.6e9, 1.54e10, 2.24e9, 2.68e9, 2.2e8, 7.8e8]
std_devs_frozen = [9.43E+07, 9.43E+08, 1.64E+09, 4.71E+07, 2.45E+08, 7.12E+07, 4.71E+07]

# Data for colony
od_values_colony = [4.33, 2.14, 1.05, 0.63, 0.40, 0.20, 0.08]
cfu_values_colony = [3e9, 2.6e9, 2.4e9, 1.06e9, 1.08e9, 1.38e9, 1.2e8]
std_devs_colony = [2.49E+08, 5.89E+08, 1.63E+08, 8.06E+07, 3.27E+07, 1.73E+08, 9.43E+06]

# Calculate regression for both datasets
slope_frozen, intercept_frozen, r_value_frozen, _, _ = stats.linregress(od_values_frozen, cfu_values_frozen)
slope_colony, intercept_colony, r_value_colony, _, _ = stats.linregress(od_values_colony, cfu_values_colony)

# Create lines of best fit
line_x = np.linspace(0, max(max(od_values_frozen), max(od_values_colony))*1.1, 100)
line_y_frozen = slope_frozen * line_x + intercept_frozen
line_y_colony = slope_colony * line_x + intercept_colony

# Plot Frozen Stock
ax1.errorbar(od_values_frozen, cfu_values_frozen, yerr=std_devs_frozen, fmt='o', 
            capsize=5, color='blue', alpha=0.5, label='Data points')
ax1.plot(line_x, line_y_frozen, color='black', label=f'Best fit line (R² = {r_value_frozen**2:.3f})')
ax1.set_yscale('linear')
ax1.set_xlabel('OD Values')
ax1.set_ylabel('CFU/ml')
ax1.set_title('OD vs CFU Count (AD38 grown from frozen stock)')
ax1.legend(loc='upper left')
equation_frozen = f'CFU = {slope_frozen:.2e}×OD + {intercept_frozen:.2e}'
ax1.text(0.95, 0.95, equation_frozen, 
         transform=ax1.transAxes,
         horizontalalignment='right',
         verticalalignment='top',
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6))

# Plot Colony
ax2.errorbar(od_values_colony, cfu_values_colony, yerr=std_devs_colony, fmt='o', 
            capsize=5, color='blue', alpha=0.5, label='Data points')
ax2.plot(line_x, line_y_colony, color='black', label=f'Best fit line (R² = {r_value_colony**2:.3f})')
ax2.set_yscale('linear')
ax2.set_xlabel('OD Values')
ax2.set_ylabel('CFU/ml')
ax2.set_title('OD vs CFU Count (AD38 grown from colony)')
ax2.legend(loc='upper left')
equation_colony = f'CFU = {slope_colony:.2e}×OD + {intercept_colony:.2e}'
ax2.text(0.95, 0.95, equation_colony, 
         transform=ax2.transAxes,
         horizontalalignment='right',
         verticalalignment='top',
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.6))

# Set the same scale for both plots
x_min = min(min(od_values_frozen), min(od_values_colony))
x_max = max(max(od_values_frozen), max(od_values_colony))
y_min = min(min(cfu_values_frozen), min(cfu_values_colony))
y_max = max(max(cfu_values_frozen), max(cfu_values_colony))

ax1.set_xlim(0, x_max*1.1)
ax2.set_xlim(0, x_max*1.1)
ax1.set_ylim(y_min/2, y_max*2)
ax2.set_ylim(y_min/2, y_max*2)

plt.tight_layout()
plt.show()

# Print statistics
print("Frozen Stock:")
print(f"Slope: {slope_frozen:.4e}")
print(f"Intercept: {intercept_frozen:.4e}")
print(f"R-squared: {r_value_frozen**2:.4f}\n")

print("Colony:")
print(f"Slope: {slope_colony:.4e}")
print(f"Intercept: {intercept_colony:.4e}")
print(f"R-squared: {r_value_colony**2:.4f}")