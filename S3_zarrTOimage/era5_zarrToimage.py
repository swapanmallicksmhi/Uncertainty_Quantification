# -*- coding: utf-8 -*-
#--  Swapan Mallick
#--  21 July 2025
#-
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import subprocess
import pandas as pd
import ast
import shutil
from PIL import Image

# Function to extract a variable from a Zarr file
def extract_variable(zarr_path, variable_name):
    try:
        ds = xr.open_zarr(zarr_path)
        if variable_name in ds:
            return ds[variable_name]
        else:
            print(f"Variable '{variable_name}' not found in {zarr_path}")
            return None
    except Exception as e:
        print(f"Error reading Zarr file: {e}")
        return None

# Function to plot the data
def plot_data(data, filename, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    data.plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap='jet',
        cbar_kwargs={'label': '', 'shrink': 0.8}
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(label)
    plt.savefig(filename, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python script.py YYYY MM \"['var1','var2']\" INPUT_DIR OUTPUT_DIR")
        sys.exit(1)

    YYYY = sys.argv[1]
    MM = sys.argv[2]
    try:
        VALID_VARIABLES = ast.literal_eval(sys.argv[3])
        if not isinstance(VALID_VARIABLES, list):
            raise ValueError
    except Exception:
        print("Error: VALID_VARIABLES must be a Python list")
        sys.exit(1)

    INPUT  = sys.argv[4]
    OUTPUT = sys.argv[5]

    # Plot styling
    plt.rcParams.update({
        'font.size': 16,
        'axes.linewidth': 1.5,
        'font.family': 'serif'
    })

    variable_mean_ranges = {
        'sp': (500, 1100),
        't2m': (200, 300), 't950': (200, 300), 't900': (200, 300),
        't700': (200, 300), 't500': (200, 300),
        'u10': (-30, 30), 'u950': (-30, 30), 'u900': (-30, 30),
        'u700': (-30, 30), 'u500': (-30, 30),
        'v10': (-30, 30), 'v950': (-30, 30), 'v900': (-30, 30),
        'v700': (-30, 30), 'v500': (-30, 30),
        'q950': (0, 10), 'q900': (0, 10), 'q700': (0, 10), 'q500': (0, 10)
    }

    variable_std_ranges = {
        'sp': (0, 2),
        't2m': (0, 3), 't950': (0, 3), 't900': (0, 3),
        't700': (0, 3), 't500': (0, 3),
        'u10': (0, 3), 'u950': (0, 3), 'u900': (0, 3),
        'u700': (0, 3), 'u500': (0, 3),
        'v10': (0, 3), 'v950': (0, 3), 'v900': (0, 3),
        'v700': (0, 3), 'v500': (0, 3),
        'q950': (0, 1), 'q900': (0, 1), 'q700': (0, 1), 'q500': (0, 1)
    }

    for VARIABLE in VALID_VARIABLES:
        print(f"\nProcessing variable: {VARIABLE}")
        output_dir = Path(f"{OUTPUT}/{VARIABLE}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_crop = Path(f"{OUTPUT}/{VARIABLE}/CROP")
        output_dir_crop.mkdir(parents=True, exist_ok=True)

        zarr_path = Path(f"{INPUT}/{YYYY}{MM}/{VARIABLE}/{VARIABLE}_ERA5_{YYYY}{MM}.zarr")

        ens_mean = extract_variable(zarr_path, f"{VARIABLE}_mean")
        ens_std = extract_variable(zarr_path, f"{VARIABLE}_std")

        if ens_mean is None or ens_std is None or 'time' not in ens_mean.coords:
            print(f"Skipping variable '{VARIABLE}' due to data issues.")
            continue

        min_en, max_en = variable_mean_ranges[VARIABLE]
        min_sd, max_sd = variable_std_ranges[VARIABLE]

        time_values = ens_mean['time'].values

        for i, time_val in enumerate(time_values):
            print(f"  Plotting time step {i + 1} - Time: {time_val}")
            time_str = str(time_val).replace(":", "").replace("-", "").replace("T", "")[:10]

            # Plot SD
            sd_filename = f"SD_{VARIABLE}_{time_str}_era5.png"
            plot_data(
                ens_std[i, :, :],
                filename=sd_filename,
                vmin=min_sd,
                vmax=max_sd,
                label=f"Standard Deviation - {time_str}"
            )

            # Plot mean
            mean_filename = f"EnsMEAN_{VARIABLE}_{time_str}_era5.png"
            plot_data(
                ens_mean[i, :, :],
                filename=mean_filename,
                vmin=min_en,
                vmax=max_en,
                label=f"Ensemble Mean - {time_str}"
            )
            #-----------Image crop---------------
            left_crop = 210; right_crop = 260
            top_crop = 80; bottom_crop = 140

            for png_file in [sd_filename, mean_filename]:
                if os.path.exists(png_file):
                    target_file = output_dir / png_file
                    shutil.copy2(png_file, target_file)
                    #-------------
                    pil_image = Image.open(png_file)
                    pil_image.load()
                    img_array = np.array(pil_image.convert("RGB"))  # Ensure 3 channels
                    height, width, channels = img_array.shape
                    cropped_img = img_array[top_crop:height - bottom_crop, left_crop:width - right_crop, :]
                    pil_image = Image.fromarray(cropped_img)
                    plt.imshow(pil_image)
                    plt.axis("off")
                    plt.savefig(png_file, bbox_inches='tight', facecolor='white', dpi=100)
                    # Move to output folder
                    target_file = output_dir_crop / png_file
                    subprocess.run(["mv", png_file, str(target_file)])

    print("All variables processed successfully.")
