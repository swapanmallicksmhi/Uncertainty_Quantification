# CARRA2 Coastline Plotting Tool

Add coastlines and geographic features to CARRA2 uncertainty field plots.

## Features

- Plots data on North Polar Stereographic projection
- Adds coastlines and country borders
- Optional latitude/longitude gridlines with labels
- Optional colorbar
- Supports three input modes:
  - **netcdf**: Upscaled netCDF files (2880x2880, grayscale normalized to [0,3])
  - **netcdf_raw**: Raw netCDF files (256x256, grayscale in [-1,1])
  - **png**: PNG files with jet colormap (inverted to recover physical values)

## Quick Start

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

This creates an environment called `coastline_plotting` with all required dependencies.

### 2. Run with the batch script

Edit `run_coastlines.sh` to configure:
- `MODE`: Input mode (`netcdf`, `netcdf_raw`, or `png`)
- `DATETIMES`: List of datetime strings to process
- `INPUT_DIR` / `OUTPUT_DIR`: Input/output directories
- `SHOW_COLORBAR`: Set to 1 to show colorbar, 0 to hide
- `SHOW_GRIDLINES`: Set to 1 to show lat/lon gridlines, 0 to hide
- `SAVE_NETCDF`: Set to 1 to export clean netCDF (PNG mode only)
- `OUTPUT_RESOLUTION`: NetCDF grid size
- `VMIN` / `VMAX`: Color scale limits

Then run:
```bash
chmod +x run_coastlines.sh
./run_coastlines.sh
```

### 3. Run for a single datetime

```bash
conda activate coastline_plotting
python add_coastlines.py 2019050100 --mode png --colorbar --gridlines
```

## Command-Line Usage

```
usage: add_coastlines.py [-h] [--input-dir INPUT_DIR] [--output-dir OUTPUT_DIR]
                         [--colorbar] [--gridlines] [--vmin VMIN] [--vmax VMAX]
                         [--mode {netcdf,netcdf_raw,png}]
                         datetime

Add coastlines to CARRA2 uncertainty field plots

positional arguments:
  datetime              DateTime string in format YYYYMMDDHH (e.g., 2019050100)

optional arguments:
  -h, --help            show this help message and exit
  --input-dir, -i       Input directory containing data files
                        (default: ../sample_data/input)
  --output-dir, -o      Output directory for PNG files
                        (default: ../sample_data/output)
  --colorbar, -c        Add colorbar to the plot
  --gridlines, -g       Add lat/lon gridlines with labels
  --vmin VMIN           Minimum value for color scale (default: 0.0)
  --vmax VMAX           Maximum value for color scale (default: 3.0)
  --mode, -m            Input mode: netcdf (upscaled), netcdf_raw (256x256),
                        or png (jet colormap) (default: netcdf)
  --save-netcdf, -n     Also save processed data as netCDF (PNG mode only)
  --output-resolution, -r
                        NetCDF output resolution: 246 (small, ~490KB),
                        2880 (full, ~64MB), or omit for native 754 (~2MB)
```

## Input Data

The script expects files in one of three formats:

| Mode | Filename Pattern | Resolution | Data Range | Border Removal |
|------|------------------|------------|------------|----------------|
| `netcdf` | `UQ_<datetime>.nc` | 2880x2880 | [0, 3] K | 57 px |
| `netcdf_raw` | `UQ_raw_<datetime>.nc` | 256x256 | [-1, 1] | 5 px |
| `png` | `UQ_<datetime>.png` | 788x788 | jet RGB | 17 px |

**Note**: The `png` mode inverts the jet colormap to recover physical values, which preserves more information than the grayscale-converted netCDF files.

Grid parameters (CARRA2 domain):
- Domain center: 45W, 84N
- Projection reference: 30W, 90N (North Pole)
- Original grid resolution: 2.5 km

## Dependencies

- Python >= 3.9
- numpy
- xarray
- netcdf4
- matplotlib
- cartopy
- pyproj
- scipy (for PNG mode jet inversion)
- pillow (for PNG mode)

## Output

- PNG files with coastlines overlaid
- NetCDF files (optional, PNG mode only) with:
  - Physical values in Kelvin (no padding)
  - Projection coordinates (x, y in meters)
  - Metadata (projection parameters, grid resolution)
