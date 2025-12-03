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




def process_era5_data(PATH_TO_DATA, VARIABLE, YY, MM, DDN, HH):

    timestamp = f"{YY+MM+DDN+HH}"
    DATE = f"{YY+MM+DDN}"
    DTG = f"{YY+MM}"

    # open dataset 
    if VARIABLE in ["t2m", "u10", "v10", "sp"]:   # surface variables
        data = xr.open_dataset(PATH_TO_DATA+f"/{DTG}/ERA5_EDA_SFC_{DATE}.nc")

        print("Minimum value for", VARIABLE, data[VARIABLE].values.min(), flush=True)
        print("Mean value for   ", VARIABLE, data[VARIABLE].values.mean(), flush=True)
        print("Maximum value for", VARIABLE, data[VARIABLE].values.max(), flush=True)

        # Variables that need scaling
        if VARIABLE in scale_factors:
            data[VARIABLE] *= scale_factors[VARIABLE]

        # Compute statistics, and flip y axis
        ensemble_mean = data[VARIABLE].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VARIABLE].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), y=data.y[::-1]).std(dim='number')

    else:   # model level variables
        data = xr.open_dataset(PATH_TO_DATA+f"/{DTG}/ERA5_EDA_ML_{DATE}.nc")
        VAR_NAME, plevel = VARIABLE[:-3], VARIABLE[-3:]
        hlevel = {"950": 123.0, "900": 119.0, "700": 106.0, "500": 96.0}
        # hlevel = {"950": 96.0, "900": 106.0, "700": 119.0, "500": 123.0}

        print("Minimum value for", VARIABLE, data[VAR_NAME].values.min(), flush=True)
        print("Mean value for   ", VARIABLE, data[VAR_NAME].values.mean(), flush=True)
        print("Maximum value for", VARIABLE, data[VAR_NAME].values.max(), flush=True)

        # Variables that need scaling
        if VARIABLE in scale_factors: data[VAR_NAME] *= scale_factors[VARIABLE]

        # Compute statistics, and flip y axis
        ensemble_mean = data[VAR_NAME].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VAR_NAME].sel(time=datetime.strptime(timestamp, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).std(dim='number')


    print("Minimum value for std of", VARIABLE, ensemble_std.values.min(), flush=True)
    print("Mean value for std of   ", VARIABLE, ensemble_std.values.mean(), flush=True)
    print("Maximum value for std of", VARIABLE, ensemble_std.values.max(), flush=True)


    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_mean": ensemble_mean,
                               VARIABLE+"_std": ensemble_std})

    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [timestamp]})


    # drop meta data variables that cause problems when merging
    if VARIABLE in ["t2m", "u10", "v10", "sp"]:   # surface variables
        ds_processed = ds_processed.drop_vars(["surface","step"])
    else:   # model level variables
        ds_processed = ds_processed.drop_vars(["hybrid","step"])


    print(f"\nERA5 variable was successfully processed: {VARIABLE}, for {timestamp}\n", flush=True)



    # Define plotting levels if available
    min_level, max_level = variable_mean_ranges[VARIABLE]
    levels = np.arange(np.floor(min_level), np.ceil(max_level), 1)
    vmin, vmax = variable_std_ranges[VARIABLE]


    # --------------------------- Save Plots --------------------------- #
    plot_sd(
        "Standard Deviation", 
        ensemble_std,
        "SD_era5.png",
        timestamp,
        vmin=vmin,
        vmax=vmax
    )

    if "q" not in VARIABLE:
        plot_ens_mean(
            "Ensemble Mean", 
            ensemble_mean,
            "EnsMEAN_era5.png",
            timestamp,
            levels
        )
    else:
        plot_sd(
            "Ensemble Mean", 
            ensemble_mean,
            "EnsMEAN_era5.png",
            timestamp,
            vmin=min_level,
            vmax=max_level
        )
    # ----------------------------------------------------------------- #


    print(f"png was generated for: {VARIABLE}, for {timestamp}\n", flush=True)

    return ds_processed

