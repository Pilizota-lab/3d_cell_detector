import numpy as np

# Organizing the data into rows
data = [
    # [OD, [Colony counts], [CFU values], Colony Avg, Given CFU/ml, Given Std Dev]
    [4.24, [59, 60, 60], [1.18E+10, 1.20E+10, 1.20E+10], 57, 1.14E+10, 9.43E+07],
    [2.09, [5, 5, 15], [1.00E+09, 1.00E+09, 3.00E+09], 8, 1.60E+09, 9.43E+08],
    [1.18, [76, 67, 87], [1.52E+10, 1.34E+10, 1.74E+10], 77, 1.54E+10, 1.64E+09],
    [0.52, [115, 110, 110], [2.30E+09, 2.20E+09, 2.20E+09], 112, 2.24E+09, 4.71E+07],
    [0.38, [125, 125, 151], [2.50E+09, 2.50E+09, 3.02E+09], 134, 2.68E+09, 2.45E+08],
    [0.22, [14, 6, 13], [2.80E+08, 1.20E+08, 2.60E+08], 11, 2.20E+08, 7.12E+07],
    [0.10, [41, 41, 36], [8.20E+08, 8.20E+08, 7.20E+08], 39, 7.80E+08, 4.71E+07]
]

print("Analysis of CFU/ml measurements and standard deviations:\n")
print("Row | OD | Colony CV% | Given Std Dev | Calculated Std Dev | Given CV% | Calculated CV%")
print("-" * 85)

for i, row in enumerate(data):
    od, colonies, cfus, avg_colony, avg_cfu, given_std = row
    
    # Calculate standard deviation and CV for colony counts
    colony_std = np.std(colonies)
    colony_cv = (colony_std/np.mean(colonies)) * 100
    
    # Calculate standard deviation and CV for CFU/ml
    calc_std = np.std(cfus)
    given_cv = (given_std/avg_cfu) * 100
    calc_cv = (calc_std/np.mean(cfus)) * 100
    
    print(f"{i+1} | {od:.2f} | {colony_cv:.1f}% | {given_std:.2e} | {calc_std:.2e} | {given_cv:.1f}% | {calc_cv:.1f}%")