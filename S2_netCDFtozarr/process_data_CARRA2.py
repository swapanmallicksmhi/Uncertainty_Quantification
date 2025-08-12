# -*- coding: utf-8 -*-
#--  Kasper Tølløse
#--  7 August 2025
#-
import os
import sys
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from datetime import datetime
from config import *



def process_carra2_data(PATH_TO_DATA, VARIABLE, YY, MM, DDN, HH):
    
    timestamp = f"{YY+MM+DDN+HH}"

    # first, create local links to all carra2 ensemble members
    for mem in range(10):  # Members from 0 to 9
            mem_IN = Path(f"{PATH_TO_DATA}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC006")     # CARRA2ENDA
            # print(f'Processing Member {mem_IN}', flush=True)
            command = ['ln', '-sf', f"{mem_IN}/{FA_name[VARIABLE]}.nc", f"FILE{mem}.nc"]
            subprocess.run(command)


    # File names
    file_names = [
        'FILE0.nc', 'FILE1.nc', 'FILE2.nc', 'FILE3.nc', 'FILE4.nc', 
        'FILE5.nc', 'FILE6.nc', 'FILE7.nc', 'FILE8.nc', 'FILE9.nc'
    ]


    # Load data
    data = xr.open_mfdataset(file_names,
                             combine='nested',
                             concat_dim='ensemble'
                             )


    # apparently, the logarithm of the pressure is stored in the FA file, so exp() is applied to get the pressure in Pa
    if VARIABLE == "sp": data[FA_name[VARIABLE]] = np.exp(data[FA_name[VARIABLE]])


    print("Minimum value for", VARIABLE, data[FA_name[VARIABLE]].values.min(), flush=True)
    print("Mean value for   ", VARIABLE, data[FA_name[VARIABLE]].values.mean(), flush=True)
    print("Maximum value for", VARIABLE, data[FA_name[VARIABLE]].values.max(), flush=True)


    # Variables that need scaling
    if VARIABLE in scale_factors: data[FA_name[VARIABLE]] *= scale_factors[VARIABLE]


    # Compute statistics
    ensemble_mean = data[FA_name[VARIABLE]].mean(dim='ensemble')
    ensemble_std = data[FA_name[VARIABLE]].std(dim='ensemble')

    print("Minimum value for std of", VARIABLE, ensemble_std.values.min(), flush=True)
    print("Mean value for std of   ", VARIABLE, ensemble_std.values.mean(), flush=True)
    print("Maximum value for std of", VARIABLE, ensemble_std.values.max(), flush=True)


    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_mean": ensemble_mean,
                               VARIABLE+"_std": ensemble_std})


    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [timestamp]})
    
    print(f"\nCARRA2 variable was successfully processed: {VARIABLE}, for {timestamp}\n", flush=True)



    # Define plotting levels if available
    min_level, max_level = variable_mean_ranges[VARIABLE]
    levels = np.arange(np.floor(min_level), np.ceil(max_level), 1)
    vmin, vmax = variable_std_ranges[VARIABLE]


    # --------------------------- Save Plots --------------------------- #
    plot_sd(
        "Standard Deviation", 
        ensemble_std,
        "SD_carra2.png",
        timestamp,
        vmin=vmin,
        vmax=vmax
    )

    if "q" not in VARIABLE:
        plot_ens_mean(
            "Ensemble Mean", 
            ensemble_mean,
            "EnsMEAN_carra2.png",
            timestamp,
            levels
        )
    else:
        plot_sd(
            "Ensemble Mean", 
            ensemble_mean,
            "EnsMEAN_carra2.png",
            timestamp,
            vmin=min_level,
            vmax=max_level
        )
    # ----------------------------------------------------------------- #

    print(f"png was generated for: {VARIABLE}, for {timestamp}\n", flush=True)

    return ds_processed

