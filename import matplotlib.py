import matplotlib.pyplot as plt

# Data for the survey
total_interviews = 9  # Total number of interviews conducted
direct_count_interest = 5  # Example number, replace with actual value
cannot_use_od = 7  # Example number, replace with actual value
use_od_want_better_measure = 2  # Example number, replace with actual value
track_yield_interest = 8 # Example number, replace with actual value

# Labels and values
questions = [
    "Interest in Direct Cell Count",
    "Cannot Use OD",
    "Use OD, Want Better Measure",
    "Interest in Tracking Yield"
]

values = [
    direct_count_interest,
    cannot_use_od,
    use_od_want_better_measure,
    track_yield_interest
]

# Create the bar chart
plt.figure(figsize=(11, 6))
bars = plt.barh(questions, values, color='skyblue')

# Add total interviews as a reference point
for index, bar in enumerate(bars):
    plt.text(values[index] + 0.5, bar.get_y() + bar.get_height() / 2, f"{values[index]} / {total_interviews}",
             va='center', ha='left', fontsize=14)

plt.xlabel("Number of Responses", fontsize=14)
plt.title(f"Interview Results (Out of {total_interviews} Interviews)", fontsize=16)

# Increase font size for y-axis labels
plt.yticks(fontsize=14)

# Adjust the margins to prevent label cut-off
plt.subplots_adjust(left=0.30, right=0.95, top=0.9, bottom=0.1)

# Show the plot
plt.show()