# -*- coding: utf-8 -*-
#--  Swapan Mallick
#--  1 January 2025
#-
import sys
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

# Check for input variable
if len(sys.argv) > 1:
    VARIABLE = sys.argv[1]
    print(f"Input Variable: {VARIABLE}")
else:
    print("Hi, please provide an input variable.")
    quit()

# File names
file_names = [
    'FILE0.nc', 'FILE1.nc', 'FILE2.nc', 'FILE3.nc', 'FILE4.nc', 
    'FILE5.nc', 'FILE6.nc', 'FILE7.nc', 'FILE8.nc', 'FILE9.nc'
]

# Load and process data
data = xr.open_mfdataset(file_names, combine='nested', concat_dim='ensemble')
ensemble_data = data[VARIABLE]

# Compute statistics
statistics = {
    "mean": ensemble_data.mean(dim='ensemble')
}

# Define dynamic levels based on the VARIABLE # THIS IS NOT IN USE SWAPAN
if VARIABLE in [ 
                'CLSVENT.ZONAL', 
                'CLSVENT.MERIDIEN', 'CLSTEMPERATURE'
                ]:
    ranges = {
        'CLSVENT.ZONAL': (-30, 30),
        'CLSVENT.MERIDIEN': (-30, 30),
        'CLSTEMPERATURE': (200, 300)
    }
    min_level, max_level = ranges[VARIABLE]
    levels = np.arange(np.floor(min_level), np.ceil(max_level) + 2, 2)
else:
    print("No levels defined for this variable. Exiting.")
    quit()

# Set global plot properties
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'

# Function to generate and save plots
def plot_statistic(stat_name, stat_data, levels, output_file):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    stat_data.plot.contourf(
        ax=ax,
        levels=levels,
        cmap='jet',
        extend='both',
        add_colorbar=True,
        cbar_kwargs={
            'label': '------',  # Degree symbol in Unicode
            'shrink': 0.8
        }
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close(fig)  # Close the figure to save memory

# Loop through statistics and create plots
for stat_name, stat_data in statistics.items():
    output_file = f"Ens{stat_name.upper()}.png"
    plot_statistic(stat_name, stat_data, levels, output_file)
    print(f"{stat_name.capitalize()} plot saved as {output_file}")

del data
# -*- coding: utf-8 -*-
#---------------------------------------------------------
#---------------------------------------------------------
# Load and process data
data = xr.open_mfdataset(file_names, combine='nested', concat_dim='ensemble')
ensemble_std = data[VARIABLE].std(dim='ensemble')
minmax_diff = data[VARIABLE].max(dim='ensemble') - data[VARIABLE].min(dim='ensemble')

# Plotting function
def plot_data(data, filename, vmin, vmax, label):
    plt.figure(figsize=(8, 6.5))
    data.plot(vmin=vmin, vmax=vmax, cmap='jet', cbar_kwargs={'label': label})
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.savefig(filename, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

# Generate and save plots
plot_data(ensemble_std, 'SD.png', vmin=0, vmax=3, label='Standard Deviation')
#------------------------------------------------------------------
quit()
