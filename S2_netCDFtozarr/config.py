import matplotlib.pyplot as plt


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


# Variables that need scaling
scale_factors = {
    'sp': 0.01,'q950': 1000.,'q900': 1000.,'q700': 1000.,'q500': 1000.
}


# Define variable-specific plotting ranges
variable_mean_ranges = {
    'sp': (500, 1100),
    't2m': (200, 300),'t950': (200, 300),'t900': (200, 300),'t700': (200, 300),'t500': (200, 300),
    'u10': (-30, 30),'u950': (-30, 30),'u900': (-30, 30),'u700': (-30, 30),'u500': (-30, 30),
    'v10': (-30, 30),'v950': (-30, 30),'v900': (-30, 30),'v700': (-30, 30),'v500': (-30, 30),
    'q950': (0, 10),'q900': (0, 10),'q700': (0, 10),'q500': (0, 10)
}

# Define variable-specific plotting ranges
variable_std_ranges = {
    'sp': (0, 2),
    't2m': (0, 3),'t950': (0, 3),'t900': (0, 3),'t700': (0, 3),'t500': (0, 3),
    'u10': (0, 3),'u950': (0, 3),'u900': (0, 3),'u700': (0, 3),'u500': (0, 3),
    'v10': (0, 3),'v950': (0, 3),'v900': (0, 3),'v700': (0, 3),'v500': (0, 3),
    'q950': (0, 1),'q900': (0, 1),'q700': (0, 1),'q500': (0, 1)
}


# Plotting style
plt.rcParams.update({
    'font.size': 16,
    'axes.linewidth': 1.5,
    'font.family': 'serif'
})



# -------- Plotting Functions -------- #

def plot_ens_mean(stat_name, stat_data, output_file, timestamp, levels):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    stat_data.plot(
        ax=ax, levels=levels, cmap='jet', extend='both',
        add_colorbar=True, cbar_kwargs={'label': '', 'shrink': 0.8}
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{stat_name} - Valid on {timestamp}")
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()

def plot_sd(stat_name, stat_data, output_file, timestamp, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    stat_data.plot(
        ax=ax, vmin=vmin, vmax=vmax, cmap='jet',
        cbar_kwargs={'label': '', 'shrink': 0.8}
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{stat_name} - Valid on {timestamp}")
    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=200)
    plt.close()
