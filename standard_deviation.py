import numpy as np

# Your diluted control data values
data = [1.40*10**8, 6.60*10**8, 1.10*10**10, 2.20*10**9, 2.16*10**10, 1.37*10**11, 1.58*10**10, 2.88*10**10]
std_devs = [5.66E+06, 5.89E+06, 3.35E+07, 7.36E+06, 1.25E+07, 1.42E+07, 7.72E+06, 2.27E+07]

mean_values = np.mean(data)
mean_std_dev = np.mean(std_devs)
cv = (mean_std_dev / mean_values) * 100

print(f"Mean value: {mean_values}")
print(f"Mean standard deviation: {mean_std_dev}")
print(f"Coefficient of Variation: {cv:.2f}%")