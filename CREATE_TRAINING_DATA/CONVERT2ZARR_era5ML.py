# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from process_data_PLOT_era5ML import process_era5_data

# === Configuration ===
SCRPDIR = "/home/swe4281/repository/CARRA_QC2025/Uncertainty_Quantification/CREATE_TRAINING_DATA"
TMPDIR = f"{SCRPDIR}/tmp_ML"
ERA_INPUT = "/scratch/fasg/CARRA2/uncert_est"
OUTPUT = f"{SCRPDIR}/OUTPUT_ERA"
PLOT_OUT = f"{OUTPUT}/PLOTS"

DTG = "202201"
DTSTR, DTEND = 5, 6  # Start and end days
YY, MM = DTG[:4], DTG[4:6]

VARIABLES_PNG = ['SD', 'EnsMEAN']
params_file = 'list_ML.txt'

# === FA Name Mapping ===

FA_name = {
    "t2m": "CLSTEMPERATURE", "q2m": "CLSHUMI.SPECIFIQ", "u10": "CLSVENT.ZONAL", "v10": "CLSVENT.MERIDIEN",
    "sp": "SURFPRESSION", "u": "u", "v": "v","q": "q"
    "t900": "S046TEMPERATURE", "t700": "S033TEMPERATURE", "t500": "S024TEMPERATURE",
    "q950": "S051HUMI.SPECIFI", "q900": "S046HUMI.SPECIFI", "q700": "S033HUMI.SPECIFI", "q500": "S024HUMI.SPECIFI",
    "u950": "S051WIND.U.PHYS", "u900": "S046WIND.U.PHYS", "u700": "S033WIND.U.PHYS", "u500": "S024WIND.U.PHYS",
    "v950": "S051WIND.V.PHYS", "v900": "S046WIND.V.PHYS", "v700": "S033WIND.V.PHYS", "v500": "S024WIND.V.PHYS"
}

# === Load Parameters ===
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# === Main Loop ===
for DD in range(DTSTR, DTEND + 1):
    DDN = f"{DD:02}"
    date_str = f"{YY}{MM}{DDN}"

    for param in list_params:
        print(f"\n--- Processing {param} for date {date_str} ---")

        input_file = Path(f"{ERA_INPUT}/ERA5_EDA_ML_{date_str}.nc")
        if not input_file.exists():
            print(f"Input file missing: {input_file}")
            continue

        os.makedirs(TMPDIR, exist_ok=True)
        os.chdir(TMPDIR)

        try:
            # Process the ERA5 ensemble data for current parameter
            process_era5_data(str(input_file), param, date_str)
            #quit()
        except Exception as e:
            print(f"Error processing {param} on {date_str}: {e}")
            continue
    #

    # Move generated plots to the output directory
        plot_output_dir = f"{PLOT_OUT}/{DTG}/{param}"
        os.makedirs(plot_output_dir, exist_ok=True)
    
        for var in VARIABLES_PNG:
            UTC_HOURS = ("00", "06", "12", "18")
            ML = ("000", "111", "222", "333")
            for utc_index, hour_str in enumerate(UTC_HOURS):
               for ml_index, ml_str in enumerate(ML):
                   filename = f"{var}_{param}_{date_str}_UTC_{utc_index}_ML_{ml_index}.png"
                   if os.path.exists(filename):
                       dest_file = f"{plot_output_dir}/{var}_{param}_{date_str}{hour_str}_ML_{ml_str}_era5.png"
                       shutil.move(filename, dest_file)
                       print(f"Moved {filename} to {dest_file}")
                   else:
                       print(f"Missing expected plot: {filename}")

            # Clean TMPDIR
        os.chdir(SCRPDIR)
        shutil.rmtree(TMPDIR, ignore_errors=True)
        print(f"Cleaned temporary directory: {TMPDIR}")

print("All processing completed.")
