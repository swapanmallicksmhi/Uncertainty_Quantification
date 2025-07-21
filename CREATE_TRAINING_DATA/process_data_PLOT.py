# -*- coding: utf-8 -*-
#-
import os
import sys
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import zarr
from datetime import datetime


def process_carra2_data(VARIABLE, FA_name, DATE):
    print(f"\nCARRA2 variable to be processed: {VARIABLE} ({FA_name}), for date {DATE}")

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

    # set down sampling method
    downsampling_option = "averaging"
    # downsampling_option = "subsampling"
    if VARIABLE in ['u10', 'v10', 't2m', 'sp', 'q950','q900','q700','q500','t950','t900','t700','t500','u950','u900','u700','u500','v950','v900','v700','v500']:
        ranges = {
                'u10': (1.),
                'v10': (1.),
                't2m': (1.),
                'sp': (100),
                'q950': (10000),'q900': (10000),'q700': (10000),'q500': (10000),
                't950': (1),'t900': (1),'t700': (1),'t500': (1),
                'u950': (1),'u900': (1),'u700': (1),'u500': (1),
                'v950': (1),'v900': (1),'v700': (1),'v500': (1)
            }
        multiply = ranges[VARIABLE]
        data[FA_name]=data[FA_name]*multiply
    else:
        print(f"No multiplecation has been done for variable '{VARIABLE}'. Exiting.")
        sys.exit(1)


    # Compute ensemble statistics
    ensemble_mean = data[FA_name].mean(dim='ensemble')
    ensemble_std = data[FA_name].std(dim='ensemble')

    if downsampling_option == "subsampling":
        # coarsen the data simply by subsampling
        ensemble_mean = ensemble_mean.sel(Y=data.Y[::12], X=data.X[::12])
        ensemble_std = ensemble_std.sel(Y=data.Y[::12], X=data.X[::12])
    elif downsampling_option == "averaging":
        # coarsen the data, i.e. calculate averages of subgrids to reduce dimensions
        ensemble_mean = ensemble_mean.coarsen(Y=12, X=12, boundary='trim').mean()
        ensemble_std = ensemble_std.coarsen(Y=12, X=12, boundary='trim').mean()

    # Rechunk the DataArray to match the dimension of the grid
    ensemble_std = ensemble_std.chunk({'Y': len(ensemble_std.Y), 'X': len(ensemble_std.X)})
    ensemble_mean = ensemble_mean.chunk({'Y': len(ensemble_mean.Y), 'X': len(ensemble_mean.X)})

    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_carra2_mean": ensemble_mean,
                               VARIABLE+"_carra2_std": ensemble_std})

    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [DATE]})
    print(f"\nCARRA2 variable was successfully processed: {VARIABLE}, for date {DATE}\n")

    # Plotting
    if True:
        ens_mean = ds_processed[VARIABLE+"_carra2_mean"]
        ens_std = ds_processed[VARIABLE+"_carra2_std"]

        # Define levels based on variable type
        if VARIABLE in ['u10', 'v10', 't2m', 'sp','t950','t900','t700','t500','u950','u900','u700','u500','v950','v900','v700','v500']:
            ranges = {
                'u10': (-30, 30),
                'v10': (-30, 30),
                't2m': (200, 300),
                'sp': (1100, 1170),
                't950': (200,300),'t900': (200,300),'t700': (200,300),'t500': (200,300),
                'u950': (-30,30),'u900': (-30,30),'u700': (-30,30),'u500': (-30,30),
                'v950': (-30,30),'v900': (-30,30),'v700': (-30,30),'v500': (-30,30)
            }
            min_level, max_level = ranges[VARIABLE]
            levels = np.arange(np.floor(min_level), np.ceil(max_level) + 2, 2)
        else:
            min_level = np.floor(ens_mean.values.min())
            max_level = np.ceil(ens_mean.values.max())
            levels = np.arange(np.floor(min_level), np.ceil(max_level) + 0.1, 0.1)
            print(f"Humidity Plots")
            #sys.exit(1)

        # Plot style settings
        plt.rcParams.update({
            'font.size': 16,
            'axes.linewidth': 1.5,
            'font.family': 'serif'
        })

        def plot_ens_mean(stat_name, stat_data, levels, output_file):
            fig, ax = plt.subplots(figsize=(8, 6.5))
            stat_data.plot(
                ax=ax, levels=levels, cmap='jet', extend='both',
                add_colorbar=True, cbar_kwargs={'label': '', 'shrink': 0.8}
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{stat_name} - Valid on {DATE}")
            plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
            plt.close()

        def plot_sd(stat_name, stat_data, vmin, vmax, output_file):
            fig, ax = plt.subplots(figsize=(8, 6.5))
            stat_data.plot(
                ax=ax, vmin=vmin, vmax=vmax, cmap='jet',
                cbar_kwargs={'label': '', 'shrink': 0.8}
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"{stat_name} - Valid on {DATE}")
            plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
            plt.close()

        plot_ens_mean("Ensemble Mean", ens_mean, levels, output_file="EnsMEAN_carra2.png")
        print('---Ensemble Mean--- Plots')
        if VARIABLE == 'sp':
            print(VARIABLE)
            plot_sd("Standard Deviation", ens_std, vmin=0, vmax=0.1, output_file="SD_carra2.png")
        else:
            print(VARIABLE)
            plot_sd("Standard Deviation", ens_std, vmin=0, vmax=3, output_file="SD_carra2.png")


        print(f"PNG files were generated for: {VARIABLE}, for date {DATE}\n")

    return ds_processed

#----------------------------------------
def process_era5_data(PATH_TO_DATA, VARIABLE, DATE):
    print(f"\nERA5 variable to be processed: {VARIABLE}, for date {DATE}")

    # open dataset
    if VARIABLE in ["t2m", "q2m", "u10", "v10", "sp", "nlwrt"]:
        data = xr.open_dataset(PATH_TO_DATA+"/ERA5_EDA_SFC_20220301.nc")

        # Compute statistics
        ensemble_mean = data[VARIABLE].sel(time=datetime.strptime(DATE, "%Y%m%d%H"), y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VARIABLE].sel(time=datetime.strptime(DATE, "%Y%m%d%H"), y=data.y[::-1]).std(dim='number')

    else:
        data = xr.open_dataset(PATH_TO_DATA+"/ERA5_EDA_ML_20220301.nc")
        VAR_NAME, plevel = VARIABLE[:-3], VARIABLE[-3:]
        hlevel = {"950": 96.0, "900": 106.0, "700": 119.0, "500": 123.0}

        # Compute statistics
        ensemble_mean = data[VAR_NAME].sel(time=datetime.strptime(DATE, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).mean(dim='number')
        ensemble_std = data[VAR_NAME].sel(time=datetime.strptime(DATE, "%Y%m%d%H"), hybrid=hlevel[plevel], y=data.y[::-1]).std(dim='number')

    # interpolate to finer grid
    new_x = np.linspace(ensemble_std.x.values[0], ensemble_std.x.values[-1], 239)
    new_y = np.linspace(ensemble_std.y.values[0], ensemble_std.y.values[-1], 239)
    ensemble_mean = ensemble_mean.interp(y=new_y, x=new_x, method="linear")
    ensemble_std = ensemble_std.interp(y=new_y, x=new_x, method="linear")

    # create new dataarrays from numpy arrays to remove metadata inconsistent with carra2 format
    coord_dict = {'Y': range(len(ensemble_std.y)), 'X': range(len(ensemble_std.x))}
    ensemble_std = xr.DataArray(ensemble_std, coords=coord_dict, dims=['Y', 'X'])
    ensemble_mean = xr.DataArray(ensemble_mean, coords=coord_dict, dims=['Y', 'X'])

    # Create new Dataset containing processed data
    ds_processed = xr.Dataset({VARIABLE+"_era5_mean": ensemble_mean,
                               VARIABLE+"_era5_std": ensemble_std})

    # expand dimension (add date, to be able to append data to exisitng zarr dataset)
    ds_processed = ds_processed.expand_dims({'time': [DATE]})

    print(f"\nERA5 variable was successfully processed: {VARIABLE}, for date {DATE}\n")


    # plot fields
    if True:
        ens_mean = ds_processed[VARIABLE+"_era5_mean"]
        ens_std = ds_processed[VARIABLE+"_era5_std"]

        # Set global plot properties
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['font.family'] = 'serif'

        # Plotting function
        def plot_data(data, filename, vmin, vmax, label):
            plt.figure(figsize=(8, 6.5))
            data.plot(vmin=vmin, vmax=vmax, cmap='jet', cbar_kwargs={'label': label})
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(filename, bbox_inches='tight', facecolor='white', dpi=200)
            plt.close()

        # Generate and save plots
        plot_data(ens_std, f"SD_era5.png", vmin=0, vmax=np.ceil(ens_std.values.max()), label='Standard Deviation')
        plot_data(ens_mean,  f"EnsMEAN_era5.png", vmin=np.floor(ens_mean.values.min()), vmax=np.ceil(ensemble_mean.values.max()), label='Ensemble Mean')
        #------------------------------------------------------------------

        print(f"png was generated for: {VARIABLE}, for date {DATE}\n")


    return ds_processed
