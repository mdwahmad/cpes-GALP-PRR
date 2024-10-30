import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Re-using the provided EV demand profile and electricity prices with the updated values
EV_demand_profile = np.array([11, 11, 22, 22, 11, 22, 44, 66, 88, 121, 90, 66, 88, 110, 90, 22, 22, 44, 55, 66, 110, 99, 22, 0])
electricity_prices = np.array([0.10, 0.10, 0.01, 0.8, 0.12, 0.01, 0.15, 0.20, 0.25, 0.30, 0.28, 0.24, 0.22, 0.20, 0.18, 0.16, 0.16, 0.18, 0.20, 0.22, 0.25, 0.24, 0.14, 0.12])

T = 24
P_max = 90
P_EV = 11
P_B_charge = 50
P_B_discharge = 50
ESS_capacity = 100
Charging_efficiency = 0.9
Discharging_efficiency = 0.9
#initial_SOC = ESS_capacity / 2
initial_SOC = 0.20 * ESS_capacity  # Changed from ESS_capacity / 2 to 0.20 * ESS_capacity
min_SOC_limit = 0.20 * ESS_capacity
max_SOC_limit = 0.95 * ESS_capacity

# Initializing arrays for Charge, Discharge, SOC, and calculating total cost
Charge = np.zeros(T)
Discharge = np.zeros(T)
SOC = np.zeros(T)
SOC[0] = initial_SOC
total_cost = 0

for t in range(T):
    EV_demand = EV_demand_profile[t]
    price = electricity_prices[t]
    available_grid_power = P_max - EV_demand
    # Print statement to monitor SOC and decisions at each hour
    print(f"Hour {t}: Initial SOC: {SOC[t-1]:.2f}, EV Demand: {EV_demand}, Price: {price}")
    
    if t == 0:
        # Skip any charging or discharging actions for hour 0
        print(f"Hour {t}: Initial setup. SOC: {SOC[t]} kWh")
    elif 0 < t <= 6:
        if SOC[t-1] < max_SOC_limit and price <= np.min(electricity_prices[:6]):
            Charge[t] = min(P_B_charge, available_grid_power, (max_SOC_limit - SOC[t-1]) / Charging_efficiency)
            SOC[t] = SOC[t-1] + Charge[t] * Charging_efficiency
            total_cost += Charge[t] * price / Charging_efficiency
            print(f"Hour {t}: Charging {Charge[t]:.2f} kW at low price of {price} EUR/kWh.")

        else:
            # If conditions not met, maintain SOC
            SOC[t] = SOC[t-1]
    elif t == 7:
        if SOC[t-1] < 0.95 * ESS_capacity:
            Charge[t] = ((0.95 * ESS_capacity) - SOC[t-1]) / Charging_efficiency
            SOC[t] = 0.95 * ESS_capacity
            total_cost += Charge[t] * price / Charging_efficiency
            print(f"Charging {Charge[t]:.2f} kW to reach 95% SOC.")
        else:
            # If already at 95% SOC, maintain SOC
            SOC[t] = SOC[t-1]
    else:
        if SOC[t-1] < max_SOC_limit and available_grid_power > 0:
            Charge[t] = min(P_B_charge, available_grid_power) * Charging_efficiency
            SOC[t] = SOC[t-1] + Charge[t] * Charging_efficiency
            print(f"Charging {Charge[t]:.2f} kW with available power.")
        elif EV_demand > P_max and SOC[t-1] > min_SOC_limit:
            Discharge[t] = min(P_B_discharge, EV_demand - P_max)
            SOC[t] = SOC[t-1] - Discharge[t] / Discharging_efficiency
            print(f"Discharging {Discharge[t]:.2f} kW due to high demand.")
        else:
            SOC[t] = SOC[t-1]

    SOC[t] = min(max_SOC_limit, max(min_SOC_limit, SOC[t]))
    print(f"Hour {t} end SOC: {SOC[t]:.2f}\n")

print(f"Total electricity cost for the day: {total_cost} EUR")


# Data preparation and DataFrame creation
data = {
    "Hour": np.arange(T),
    "Charge (kW)": Charge,
    "Discharge (kW)": Discharge,
    "SOC (kWh)": SOC,
    "EV Demand (kW)": EV_demand_profile,
    "Cars Charging": EV_demand_profile // P_EV,
    "Battery Support": ["Yes" if d > 0 else "No" for d in Discharge],
    "Charging Event": ["Yes" if c > 0 else "No" for c in Charge],
    "Discharging Event": ["Yes" if d > 0 else "No" for d in Discharge]
}

# Convert to DataFrame for pretty table display
df = pd.DataFrame(data)

# Print table
print(df.to_string(index=False))


################################
################################


# DataFrame 'df' contains the simulation results as per the provided code
hours = df['Hour']
ev_demand = df['EV Demand (kW)']
charge = df['Charge (kW)']
discharge = df['Discharge (kW)']
soc = df['SOC (kWh)']
cars_charging = df['Cars Charging']

# Increase figure size for better visibility
fig, ax1 = plt.subplots(figsize=(20, 12))  # Adjusted figure size for better visibility

# Plot EV Demand
ax1.plot(hours, ev_demand, label='EV Demand (kW)', color='blue', marker='o', linestyle='-')
ax1.set_xlabel('Hour', fontsize=14)  # Increased font size
ax1.set_ylabel('EV Demand (kW) / SOC (kWh)', color='blue', fontsize=14)  # Increased font size
ax1.tick_params(axis='y', labelcolor='blue', labelsize=12)  # Increased tick label size

# Adding SOC to the same axis
ax1.plot(hours, soc, label='SOC (kWh)', color='red', linestyle='--')
ax1.fill_between(hours, 0, soc, color='red', alpha=0.1)  # Fill under SOC curve

# Instantiate a second y-axis to plot charging activities and number of cars charging
ax2 = ax1.twinx()
ax2.set_ylabel('Charging/Discharging (kW) / Cars Charging', color='green', fontsize=14)  # Increased font size
ax2.step(hours, charge, label='Charging (kW)', color='green', linestyle='-', where='post')
ax2.step(hours, -discharge, label='Discharging (kW)', color='orange', linestyle='-', where='post')  # Negative for visual distinction
ax2.bar(hours, cars_charging, color='grey', alpha=0.3, label='Cars Charging')
ax2.tick_params(axis='y', labelcolor='green', labelsize=12)  # Increased tick label size

# Adding legends to the plot
ax1.legend(loc='upper left', fontsize=12)  # Increased legend font size
ax2.legend(loc='upper right', fontsize=12)  # Increased legend font size

plt.show()


#######################
#######################
# Lighter colors for better distinction
colors = {
    'EV Demand': '#add8e6',  # Light blue
    'Charging': '#98fb98',  # Pale green
    'Discharging': '#ffb6c1',  # Light pink
    'SOC': '#fdfd96',  # Light yellow
    'Cars Charging': '#724db8'  # Light purple
}

# Specified periods for visualization
specified_periods = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 20, 21]

# Assuming 'df' is your DataFrame containing the simulation data
filtered_ev_demand = df['EV Demand (kW)'][specified_periods].values
filtered_charge = df['Charge (kW)'][specified_periods].values
filtered_discharge = df['Discharge (kW)'][specified_periods].values
filtered_soc = df['SOC (kWh)'][specified_periods].values
filtered_cars_charging = df['Cars Charging'][specified_periods].values

# Setup for the plot
fig, ax = plt.subplots(figsize=(18, 10))  # Increased plot size

# Bar width
bar_width = 0.15

# Positions of the bar groups
r1 = np.arange(len(specified_periods))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]
r4 = [x + bar_width for x in r3]
r5 = [x + bar_width for x in r4]

# Plotting bars with specified colors
ax.bar(r1, filtered_ev_demand, color=colors['EV Demand'], width=bar_width, edgecolor='grey', label='EV Demand (kW)')
ax.bar(r2, filtered_charge, color=colors['Charging'], width=bar_width, edgecolor='grey', label='Charging (kW)')
ax.bar(r3, filtered_discharge, color=colors['Discharging'], width=bar_width, edgecolor='grey', label='Discharging (kW)')
ax.bar(r4, filtered_soc, color=colors['SOC'], width=bar_width, edgecolor='grey', label='SOC (kWh)')
ax.bar(r5, filtered_cars_charging, color=colors['Cars Charging'], width=bar_width, edgecolor='grey', label='Cars Charging')

# Adding values on top of each bar
def add_values(ax, data, positions):
    for pos, value in zip(positions, data):
        ax.text(pos, value + 1, f'{value:.1f}', ha='center', va='bottom', fontsize=10, color='black', rotation=90, fontweight='bold')  # Make values bold

# Call the function to add values
add_values(ax, filtered_ev_demand, r1)
add_values(ax, filtered_charge, r2)
add_values(ax, filtered_discharge, r3)
add_values(ax, filtered_soc, r4)
add_values(ax, filtered_cars_charging, r5)

# Add the P_max line as an asymptote
ax.axhline(y=P_max, color='red', linestyle='--', linewidth=2, label=f'Max Power Capacity ({P_max} kW)')

# Adjust y-axis limit if necessary
ax.set_ylim(0, max(np.max(filtered_ev_demand), np.max(filtered_charge), np.max(filtered_discharge), np.max(filtered_soc), np.max(filtered_cars_charging), P_max) + 10)  # Extending y-axis

# Final plot adjustments
ax.set_xlabel('Hour of the Day', fontsize=14)
ax.set_ylabel('Values', fontsize=14)
#ax.set_title('Battery and EV Charging Dynamics for Selected Periods', fontsize=16)
ax.set_xticks([r + 2*bar_width for r in range(len(specified_periods))])
ax.set_xticklabels(specified_periods, fontsize=12)
ax.legend()

plt.tight_layout()
plt.show()
