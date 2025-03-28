import numpy as np

# Constants from the equation you provided
intercept = -4.9032
slope = 0.6783

# Function to predict CFU/mL given an OD
def predict_cfu(od_input):
    predicted_log_cfu = (od_input - intercept) / slope
    predicted_cfu = 10**predicted_log_cfu  # Convert back from log10 to CFU/mL
    return predicted_cfu

# Function to predict OD given a CFU/mL
def predict_od(cfu_input):
    predicted_log_cfu = np.log10(cfu_input)
    predicted_od = slope * predicted_log_cfu + intercept
    return predicted_od

# User prompt for input choice
choice = input("Do you want to input OD or CFU/mL? Type 'OD' or 'CFU': ").strip().upper()

if choice == 'OD':
    od_input = float(input("Enter the OD value: "))
    predicted_cfu = predict_cfu(od_input)
    print(f"Predicted CFU/mL for OD = {od_input}: {predicted_cfu:.2e} CFU/mL")
elif choice == 'CFU':
    cfu_input = float(input("Enter the CFU/mL value: "))
    predicted_od = predict_od(cfu_input)
    print(f"Predicted OD for CFU/mL = {cfu_input:.2e}: {predicted_od:.4f}")
else:
    print("Invalid input. Please type 'OD' or 'CFU'.")
