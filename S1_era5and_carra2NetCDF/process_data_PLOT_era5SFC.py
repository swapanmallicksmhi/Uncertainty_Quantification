# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from datetime import datetime

def process_era5_data(path_to_data, VARIABLE, date):
    """
    Process ERA5 ensemble data for a given variable and date.

    Args:
        path_to_data (str): Path to the NetCDF ERA5 file.
        VARIABLE (str): Variable name (e.g., 't2m', 'u10', etc.).
        date (str): Date string (e.g., '20220106') used for labeling.

    Outputs:
        PNG plots of ensemble mean and standard deviation for each time step.
    """

    print(f"\nProcessing ERA5 variable: {VARIABLE}, for date: {date}")

    try:
        # Load dataset
        ds = xr.open_dataset(path_to_data)
        print(f"Dataset loaded: {path_to_data}")

        # Variables that need scaling
        scale_factors = {
            'u10': 1.0, 'v10': 1.0, 't2m': 1.0, 'sp': 0.01,
            't': 1.0, 'q': 10000, 'u': 1.0, 'v': 1

        }

        if VARIABLE in scale_factors:
            multiply = scale_factors[VARIABLE]
            if VARIABLE == 'sp':
                print("Before scaling SP:", np.max(ds[VARIABLE]), np.min(ds[VARIABLE]))
            ds[VARIABLE] = ds[VARIABLE] * multiply
        else:
            print(f"No multiplication factor found for variable '{VARIABLE}'. Exiting.")
            sys.exit(1)
        # Flip latitude (y-axis) if needed
        ds_flipped = ds.reindex(y=ds.y[::-1])
        print("Latitude flipped (north to south)")

        # Plotting style
        plt.rcParams.update({
            'font.size': 16,
            'axes.linewidth': 1.5,
            'font.family': 'serif'
        })

        # Define variable-specific plotting ranges
        variable_ranges = {
            'u10': (-30, 30), 'v10': (-30, 30), 't2m': (200, 300), 'sp': (500, 1170),
            't': (200, 300), 'u': (-30, 30), 'v': (-30, 30),'q': (-30, 30)
        }

        # Loop through each time step
        for i, timestamp in enumerate(ds_flipped.time.values):
            timestamp_str = np.datetime_as_string(timestamp, unit='s')
            YYYY, MM, DD = timestamp_str[:4], timestamp_str[5:7], timestamp_str[8:10]
            UTC_hour = np.datetime_as_string(timestamp, unit='h')[-2:]

            print(f"Processing time step {i}: {timestamp_str}")

            # Extract data for current time
            t_slice = ds_flipped[VARIABLE].sel(time=timestamp)

            # Compute ensemble statistics
            ensemble_mean = t_slice.mean(dim='number')
            ensemble_std = t_slice.std(dim='number')

            # Optional: interpolate to finer grid (e.g., 239x239)
            new_x = np.linspace(ensemble_std.x.values[0], ensemble_std.x.values[-1], 239)
            new_y = np.linspace(ensemble_std.y.values[0], ensemble_std.y.values[-1], 239)

            ens_mean = ensemble_mean.interp(y=new_y, x=new_x, method="linear")
            ens_std = ensemble_std.interp(y=new_y, x=new_x, method="linear")

            # Define plotting levels if available
            if VARIABLE in variable_ranges:
                min_level, max_level = variable_ranges[VARIABLE]
                levels = np.arange(np.floor(min_level), np.ceil(max_level) + 2, 2)
            else:
                min_level = float(np.floor(ens_mean.values.min()))
                max_level = float(np.ceil(ens_mean.values.max()))
                levels = np.linspace(min_level, max_level, num=10)

            # -------- Plotting Functions -------- #
            def plot_ens(data, filename, levels, label):
                plt.figure(figsize=(8, 6.5))
                data.plot(
                    levels=levels, cmap='jet', extend='both',
                    #cmap='jet', extend='both',
                    cbar_kwargs={'label': label, 'shrink': 0.8}
                )
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title(label + f" (UTC {np.datetime_as_string(timestamp, unit='h')})")
                plt.savefig(filename, bbox_inches='tight', facecolor='white', dpi=200)
                plt.close()
            #
            def plot_sd(data, filename, vmin, vmax, label):
                plt.figure(figsize=(8, 6.5))
                data.plot(
                    vmin=vmin, vmax=vmax, cmap='jet',
                    cbar_kwargs={'label': label, 'shrink': 0.8}
                )
                plt.xlabel("Longitude")
                plt.ylabel("Latitude")
                plt.title(label + f" (UTC {np.datetime_as_string(timestamp, unit='h')})")
                plt.savefig(filename, bbox_inches='tight', facecolor='white', dpi=200)
                plt.close()

            # -------- Save Plots -------- #
            plot_sd(
                ens_std,
                f"SD_{VARIABLE}_{YYYY}{MM}{DD}_UTC_{i}.png",
                vmin=0,
                vmax=3,
                #vmax=np.ceil(ens_std.values.max()),
                label='Standard Deviation'
            )

            plot_ens(
                ens_mean,
                f"EnsMEAN_{VARIABLE}_{YYYY}{MM}{DD}_UTC_{i}.png",
                #vmin=np.floor(ens_mean.values.min()),
                #vmax=np.ceil(ens_mean.values.max()),
                levels,
                label='Ensemble Mean'
            )

            print(f"Plots saved for time step {i}")

    except Exception as e:
        print("Error during ERA5 data processing:")
        import traceback
        traceback.print_exc()
