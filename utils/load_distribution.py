import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates

# Pattern to match your CSV files
file_pattern = "data/load_profiles/*.csv"
file_paths = glob.glob(file_pattern)

# Initialize an empty DataFrame for the combined data
combined_df = pd.DataFrame()

# Iterate over the file paths
for i, file_path in enumerate(file_paths):
    # Read the current CSV file
    temp_df = pd.read_csv(file_path)

    # Correct the '24:00:00' time strings and prepend a base date for conversion
    temp_df["time"] = temp_df["time"].str.replace("24:00:00", "00:00:00")
    base_date = "2024-01-01 "
    temp_df["time"] = base_date + temp_df["time"]

    # # Convert 'time' to datetime
    temp_df["datetime"] = pd.to_datetime(temp_df["time"], format="%Y-%m-%d %H:%M:%S")

    # Set 'datetime' as the index
    temp_df.set_index("datetime", inplace=True)

    # Drop the original 'time' column
    temp_df.drop("time", axis=1, inplace=True)

    # Moving last value to first row
    last_row = temp_df.iloc[-1:]
    last_row["mult"] = temp_df["mult"].iloc[0]
    temp_df = pd.concat([last_row, temp_df.iloc[:-1]], ignore_index=False)

    # Rename 'mult' to a unique name using the file index or name
    column_name = f"profile_{i+1}"
    temp_df.rename(columns={"mult": column_name}, inplace=True)

    # If it's the first file, initialize combined_df with this data
    if combined_df.empty:
        combined_df = temp_df
    else:
        combined_df = combined_df.join(temp_df, how="outer")

# Identify the Maximum Value in Each Column
max_values = combined_df.max()
combined_df.index = pd.to_datetime(combined_df.index)

# Calculate the sum of energy consumption across all profiles for each timestamp
total_energy_consumption = combined_df.sum(axis=1)

# Step 2: Normalize Each Column by Maximum power value
normalized_df = combined_df.divide(total_energy_consumption.max() / 100, axis="columns")

# Step 3: Calculate the Average Profile
average_profile = normalized_df.mean(axis=1)

# Step 4: Apply a rolling mean to smooth the data
rolling_window = 60  # for example, a 10 minute rolling window
smoothed_total_energy_consumption = average_profile.rolling(
    window=rolling_window, min_periods=1
).mean()

# Step 5: Calculate the seasonal adjustments
autumn_adjustment = smoothed_total_energy_consumption * 0.80
winter_adjustment = smoothed_total_energy_consumption 
spring_adjustment = smoothed_total_energy_consumption * 0.85
summer_adjustment = smoothed_total_energy_consumption * 0.65

seasons = pd.DataFrame(
    {
        "winter": winter_adjustment,
        "spring": spring_adjustment,
        "summer": summer_adjustment,
        "autumn": autumn_adjustment,
    }
)
seasons.to_csv("data/load_seasons.csv")
# Set the figure size and resolution
plt.figure(figsize=(8, 4), dpi=300)
# Set the plot font to Arial, which is a sans-serif font
plt.rc("font", family="Arial")
# Plotting the seasonal adjustments
winter_adjustment.plot(label="Winter", lw=2)
autumn_adjustment.plot(label="Autumn", lw=2)
summer_adjustment.plot(label="Summer", lw=2)
spring_adjustment.plot(label="Spring", lw=2)
plt.xticks([])

# Labeling the axes with a larger font for clarity
plt.xlabel("Time (H)", fontsize=12)
plt.ylabel("Total Power Consumption (kW)", fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
# Save the plot as a high-resolution PNG file
plt.savefig("plots/IEEE_formatted_plot.png", format="png")
plt.show()

