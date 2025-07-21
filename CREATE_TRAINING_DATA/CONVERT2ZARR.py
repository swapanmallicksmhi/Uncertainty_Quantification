import os
import sys
import subprocess
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from process_data_PLOT import process_carra2_data, process_era5_data


# Modify the input andoutput directories
SCRPDIR = "/home/swe4281/repository/CARRA_QC2025/Uncertainty_Quantification/CREATE_TRAINING_DATA1"
TMPDIR = f"{SCRPDIR}/tmp_ERA_C1"
INPUT1 = f"/lus/h2resw01/scratch/swe4281/DDPM_DATA/KESP/NCFILES"
OUTPUT=f"/ec/res4/hpcperm/swe4281/DDPM_DATA/FINAL_2025"
PLOT_OUT = f"{OUTPUT}/PLOTS"


# Input Date Configuration
DTG = "202209"
DTSTR, DTEND = 15, 30
YY, MM = DTG[:4], DTG[4:6]
output_file = f"{OUTPUT}/2_{YY}{MM}_1.zarr"

os.makedirs(PLOT_OUT, exist_ok=True)
VARIABLES_PNG = ['SD', 'EnsMEAN']


# Ensemble Member Configuration
ens_mem = range(10)  # Members from 0 to 9


# Load Parameters from External File
params_file = 'list_examples.txt'
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]


# list of FA names corresponding to variable names in era5 netcdf files
FA_name = {"t2m": "CLSTEMPERATURE",
           "q2m": "CLSHUMI.SPECIFIQ",
           "u10": "CLSVENT.ZONAL",
           "v10": "CLSVENT.MERIDIEN",
           "sp": "SURFPRESSION",
           "nlwrt": "SOMMRAYT.TERREST",
           "t950": "S051TEMPERATURE",
           "t900": "S046TEMPERATURE",
           "t700": "S033TEMPERATURE",
           "t500": "S024TEMPERATURE",
           "q950": "S051HUMI.SPECIFI",
           "q900": "S046HUMI.SPECIFI",
           "q700": "S033HUMI.SPECIFI",
           "q500": "S024HUMI.SPECIFI",
           "u950": "S051WIND.U.PHYS",
           "u900": "S046WIND.U.PHYS",
           "u700": "S033WIND.U.PHYS",
           "u500": "S024WIND.U.PHYS",
           "v950": "S051WIND.V.PHYS",
           "v900": "S046WIND.V.PHYS",
           "v700": "S033WIND.V.PHYS",
           "v500": "S024WIND.V.PHYS"}

# Main Processing Loop
for DD in range(DTSTR, DTEND+1):  # Adjust this range if needed
    DDN = f"{DD:02}"
    for HH in ['00', '06', '12', '18']:
        timestamp = f"{DTG+DDN+HH}"

        # If the file already exists, check for the date
        if os.path.exists(output_file):
            print("\nfile exists")
            existing_ds = xr.open_zarr(output_file)
            # check if variable exists in dataset already
            if timestamp in existing_ds['time'].values:
                print(f"Date {timestamp} already processed. Skipping.")
                continue

        # initialize empty dataset
        ds_datetime = xr.Dataset()

        # loop over parameters
        for param in list_params:
            #
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)
            for mem in ens_mem:
                 mem_IN = Path(f"{INPUT1}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC006")     # CARRA2ENDA
                 print(f'Processing Member {mem_IN}', flush=True)
                 command = ['ln', '-sf', f"{mem_IN}/{FA_name[param]}.nc", f"FILE{mem}.nc"]
                 subprocess.run(command)
                 
            # process data
            ds_param = process_carra2_data(param, FA_name[param], timestamp)
            print("carra2", ds_param)

            # merge datasets
            ds_datetime = xr.merge([ds_datetime, ds_param])

            # process data
            ###ds_param = process_era5_data(INPUT2, param, timestamp)
            # print("era5", ds_param)

            # merge datasets
            ds_datetime = xr.merge([ds_datetime, ds_param])

            # move figures---
            OUTPUT=f"{PLOT_OUT}/{DTG}/{param}"
            os.makedirs(OUTPUT, exist_ok=True)
            for png1 in VARIABLES_PNG:
                if os.path.exists(f"{png1}_carra2.png"):
                    file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_carra2.png"
                    mvfile = ['mv', f"{png1}_carra2.png", f"{file_out}"]
                    subprocess.run(mvfile)
                if os.path.exists(f"{png1}_era5.png"):
                    file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_era5.png"
                    mvfile = ['mv', f"{png1}_era5.png", f"{file_out}"]
                    subprocess.run(mvfile)

            # delete TMPDIR---
            os.chdir(SCRPDIR)
            subprocess.run(["rm", "-rf", TMPDIR], cwd=SCRPDIR)

        # write dataset to zarr archive
        if os.path.exists(output_file):
            # write to zarr
            ds_datetime.to_zarr(output_file, mode='a', append_dim="time")
        else:
            ds_datetime.to_zarr(output_file, mode='w')
        print(f"Data for {timestamp} successfully written to zarr database.", flush=True)

        print("Current content of zarr archive:\n", xr.open_dataset(output_file), flush=True)
    #  HH

print("Final content of zarr archive:\n", xr.open_dataset(output_file), flush=True)
quit()
