import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Example data for OD and CFU/mL counts
cfu_per_ml = np.array([4e6, 6.3e7, 2.22e8])  # Replace with your CFU/mL data
od_values = np.array([0.067, 0.500, 0.800])  # Replace with your OD values

# Reshape data for linear regression
cfu_per_ml_log = np.log10(cfu_per_ml).reshape(-1, 1)  # Using log10 to linearize the relationship
od_values = od_values.reshape(-1, 1)

# Fit a linear regression model
regressor = LinearRegression()
regressor.fit(cfu_per_ml_log, od_values)

# Generate projected OD values for a range of CFU/mL counts starting from 1e8
cfu_range = np.logspace(6, 12, num=100)  # Example range for CFU/mL (1e8 to 1e12)
cfu_range_log = np.log10(cfu_range).reshape(-1, 1)
projected_od = regressor.predict(cfu_range_log)

# Plotting the standard curve
plt.figure(figsize=(8, 6))
plt.scatter(cfu_per_ml, od_values, color='red', label='Observed data')
plt.plot(cfu_range, projected_od, color='blue', label='Fitted line')

# Labeling the plot
plt.xscale('log')
plt.xlabel('CFU/mL')
plt.ylabel('OD')
plt.title('Standard Curve of OD vs CFU/mL (Starting from OD = 0.067)')
plt.legend()
plt.grid(True)
plt.show()

# Print the equation of the line
slope = regressor.coef_[0][0]
intercept = regressor.intercept_[0]
print(f"The equation is: log10(CFU/mL) = (OD - {intercept:.4f}) / {slope:.4f}")

# To predict CFU/mL given an OD, you use the rearranged equation:
def predict_cfu(od_input):
    predicted_log_cfu = (od_input - intercept) / slope
    predicted_cfu = 10**predicted_log_cfu  # Convert back from log10 to CFU/mL
    return predicted_cfu

# Example projection
input_od = 0.016  # Replace with the OD you want to use for prediction
predicted_cfu = predict_cfu(input_od)
print(f"Predicted CFU/mL for OD = {input_od}: {predicted_cfu:.2e} CFU/mL")

