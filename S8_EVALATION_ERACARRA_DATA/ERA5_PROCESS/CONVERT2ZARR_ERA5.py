# -*- coding: utf-8 -*-
#--  Kasper Tølløse
#--  7 August 2025
#-
import os
import sys
import subprocess
import shutil
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from process_data_ERA5 import process_era5_data
from config import *



# Parse arguments from command line
params_list = sys.argv[1]             # parameter list file
DTG = sys.argv[2]                     # Input Date (year and month)
DTSTR = int(sys.argv[3])              # Input Date (start date)
DTEND = int(sys.argv[4])              # Input Date (end date)
YY, MM = DTG[:4], DTG[4:6]
INPUT = (sys.argv[5])
OUTDIR = (sys.argv[6])

MODEL = "ERA5"

# Modify the input and output directories
SCRPDIR = "/home/swe4281/repository/CARRA_QC2025/Uncertainty_Quantification_git/S2_netCDFtozarr"  # directory of conversion script
TMPDIR = f"{OUTDIR}/tmp_{MODEL}_{params_list}_{DTG}_{DTSTR}_{DTEND}"      # working directory
#OUTDIR = f"/ec/res4/scratch/dnk8136/DDPM_DATA/"                            # path to output
#INPUT = f"/scratch/fasg/CARRA2/uncert_est/"                               # path to ERA5 netcdf files
#INPUT = f"/scratch/fasg/CARRA2/uncert_est/"                               # path to ERA5 netcdf files


# Load Parameters
if ".txt" in params_list:
    if not os.path.exists(params_list):
        raise FileNotFoundError(f"Parameter file '{params_list}' not found.")
    with open(params_list, 'r') as f:
        params = [line.strip() for line in f if line.strip()]
else:
    params = [s.strip() for s in params_list.split(',')]


# create directories
os.makedirs(f"{OUTDIR}", exist_ok=True)
os.makedirs(f"{OUTDIR}/PLOTS", exist_ok=True)
os.makedirs(f"{OUTDIR}/MLDATA", exist_ok=True)
os.makedirs(f"{OUTDIR}/MLDATA/{DTG}", exist_ok=True)


# list of errors
errors =  []


# Main Processing Loop
for DD in range(DTSTR, DTEND+1):  # loop over days
    DDN = f"{DD:02}"

    for HH in ['00', '06', '12', '18']:  # loop over hours

        timestamp = f"{DTG+DDN+HH}"

        for param in params:  # loop over parameters

            # create folder for variable
            os.makedirs(f"{OUTDIR}/MLDATA/{DTG}/{param}", exist_ok=True)
            ZARROUT = f"{OUTDIR}/MLDATA/{DTG}/{param}"

            # If the file already exists, check for the date
            if os.path.exists(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr" ):
                existing_ds = xr.open_zarr(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr" )
                # check if variable exists in dataset already
                if timestamp in existing_ds['time'].values:
                    print(f"Date {timestamp} already processed. Skipping.", flush=True)
                    continue


            # create and change to temp dir
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)


            print(f"\nERA5 variable to be processed: {param}, for {timestamp}", flush=True)


            try:

                # process CARRA2 data (for param)
                ds_param = process_era5_data(INPUT,
                                            param,
                                            YY, MM, DDN, HH)
                print("Single time step dataset:", ds_param, flush=True)


                # move figures---
                OUTPUT=f"{OUTDIR}/PLOTS/{DTG}/{param}"
                os.makedirs(OUTPUT, exist_ok=True)
                for png1 in ['SD', 'EnsMEAN']:
                    if os.path.exists(f"{png1}_era5.png"):
                        file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_era5.png"
                        mvfile = ['mv', f"{png1}_era5.png", f"{file_out}"]
                        subprocess.run(mvfile)


                # write dataset to zarr archive
                if os.path.exists(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr" ):
                    # write to zarr
                    print("\nfile '{MODEL}/ZARRFILES/{param}.zarr' exists. Data is appended to existing archive.", flush=True)
                    ds_param.to_zarr(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr" , mode='a', append_dim="time")
                else:
                    print("\nfile '{MODEL}/ZARRFILES/{param}.zarr' does not exists. New archive is created.", flush=True)
                    ds_param.to_zarr(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr" , mode='w')
                print(f"Data for {timestamp} successfully written to zarr database.", flush=True)

                print("Current content of zarr archive:\n", xr.open_dataset(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr"), flush=True)

            except Exception as e:
                error_msg = f"For {timestamp}, parameter {param} was not processed due to error: {str(e)}"
                print(error_msg, flush=True)
                errors.append(error_msg)


            # delete TMPDIR---
            os.chdir(SCRPDIR)
            subprocess.run(["rm", "-rfd", TMPDIR], cwd=SCRPDIR)

    #  HH
# At the end, print all errors
if errors:
    print("\n\n\nSummary of errors:\n", flush=True)
    for err in errors:
        print(err, flush=True)
    print("\n\n\n", flush=True)
else:
    print("\n\n\nSummary:\n", flush=True)
    print("All data processed successfully.")
    print("\n\n\n", flush=True)

print("Final content of zarr archive:\n", xr.open_dataset(f"{ZARROUT}/{param}_{MODEL}_{DTG}.zarr"), flush=True)
quit()
