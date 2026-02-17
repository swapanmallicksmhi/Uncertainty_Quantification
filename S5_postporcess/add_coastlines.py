# -*- coding: utf-8 -*-
"""
Add coastlines to CARRA2 uncertainty field plots using cartopy.

Supports three input modes:
- netcdf: Upscaled netCDF files (2880x2880, grayscale normalized to [0,3])
- netcdf_raw: Raw netCDF files (256x256, grayscale in [-1,1])
- png: PNG files with jet colormap (inverted to get true physical values)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from PIL import Image
from matplotlib import cm
from scipy.spatial import cKDTree

# CARRA2 base parameters (North Polar Stereographic)
# Original model config: NLON=2880, NLAT=2880, LONC=-45, LATC=84, LON0=-30, LAT0=90, GSIZE=2500
CARRA2_BASE_NX = 2880
CARRA2_BASE_DX = 2500.0  # meters


def get_carra2_params(input_size=None, border=0):
    """
    Get CARRA2 grid parameters for a given input size and border.

    Parameters
    ----------
    input_size : int or None
        Input grid size (e.g., 2880, 256, 788). If None, returns base CARRA2 params.
    border : int or None
        Border size to remove from each side. If None, no border removal.

    Returns
    -------
    params : dict
        Grid parameters including projection info, resolution, and grid dimensions.

    Examples
    --------
    >>> get_carra2_params()                    # Base CARRA2: 2880x2880, dx=2500m
    >>> get_carra2_params(2880, 57)            # Upscaled netCDF: 2766x2766
    >>> get_carra2_params(256, 5)              # Raw netCDF: 246x246
    >>> get_carra2_params(788, 17)             # PNG: 754x754
    """
    # Static projection parameters
    params = {
        'central_longitude': -30.0,   # LON0 - projection reference longitude
        'true_scale_latitude': 90.0,  # LAT0 - North Pole
        'domain_center_lon': -45.0,   # LONC
        'domain_center_lat': 84.0,    # LATC
    }

    # Calculate grid dimensions
    if input_size is None:
        # Base CARRA2 parameters
        nx = ny = CARRA2_BASE_NX
        dx = CARRA2_BASE_DX
    else:
        # Calculate grid size after border removal
        nx = ny = input_size - 2 * border
        # Grid resolution scales inversely with grid points
        # (total domain extent is fixed at 2880 * 2500 m)
        dx = CARRA2_BASE_DX * CARRA2_BASE_NX / nx

    params['nx'] = nx
    params['ny'] = ny
    params['grid_resolution'] = dx

    return params


def read_netcdf(nc_path):
    """
    Read netCDF file and return data as numpy array.

    The UQ netCDF files contain a single data variable (the uncertainty field).
    This function extracts that variable and returns it as a 2D numpy array.
    """
    ds = xr.open_dataset(nc_path)

    # Print dataset summary for debugging
    print("=== Reading netCDF ===")
    var_name = list(ds.data_vars)[0]
    da = ds[var_name]
    print(f"Variable: {var_name}, shape: {da.shape}")
    print(f"Range: [{float(da.min()):.4f}, {float(da.max()):.4f}]")

    return da.values


def read_png(png_path, vmin=0.0, vmax=3.0):
    """
    Read PNG file and invert jet colormap to get physical values.

    Parameters
    ----------
    png_path : str
        Path to PNG file
    vmin, vmax : float
        The colorbar range used when creating the PNG

    Returns
    -------
    data : array-like
        Physical values array (same shape as input PNG, flipped to origin='lower')
    """
    img = Image.open(png_path)
    img_array = np.array(img)[:, :, :3]  # Get RGB, ignore alpha

    print(f"=== Reading PNG ===")
    print(f"Total shape: {img_array.shape}")

    # Invert jet colormap
    physical, mask = invert_jet_rgb(img_array, vmin, vmax)

    # Analyze physical values in valid data region
    valid_data = physical[mask]
    print(f"Physical range: [{np.nanmin(valid_data):.4f}, {np.nanmax(valid_data):.4f}]")

    # Flip vertically: PNG has origin at top, we need origin at bottom
    physical = np.flipud(physical)

    return physical


# Pre-compute jet colormap lookup table for RGB inversion
_JET_SCALARS = np.linspace(0, 1, 1000)
_JET_RGB = (cm.get_cmap('jet')(_JET_SCALARS)[:, :3] * 255).astype(np.uint8)
_JET_TREE = cKDTree(_JET_RGB)


def invert_jet_rgb(rgb_array, vmin=0.0, vmax=3.0):
    """
    Invert jet colormap RGB values to physical values.

    Parameters
    ----------
    rgb_array : array-like
        RGB array of shape (H, W, 3) with values in [0, 255]
    vmin, vmax : float
        The original colorbar range used when creating the PNG

    Returns
    -------
    physical : array-like
        Physical values in [vmin, vmax] range
    mask : array-like
        Boolean mask where True = valid jet color, False = background
    """
    flat_rgb = rgb_array.reshape(-1, 3)
    distances, indices = _JET_TREE.query(flat_rgb)

    # Get scalar values [0, 1] and scale to physical range
    scalars = _JET_SCALARS[indices]
    physical = vmin + scalars * (vmax - vmin)
    physical = physical.reshape(rgb_array.shape[:2])

    # Mask out non-jet colors (background) based on RGB distance
    mask = distances.reshape(rgb_array.shape[:2]) < 50

    return physical, mask


def process_data(data, border=5, scale=True):
    """
    Process diffusion model output by removing border and optionally scaling.

    Parameters
    ----------
    data : array-like
        Input data array
    border : int
        Border size to remove in pixels (5 for raw 256x256, 57 for upscaled 2880x2880)
    scale : bool
        If True, denormalize from [-1, 1] to [0, 3] K (for raw data)
        If False, data is already in physical units (for upscaled data)
    """
    print(f"\n=== Processing Data ===")
    print(f"Input shape: {data.shape}, range: [{np.nanmin(data):.4f}, {np.nanmax(data):.4f}]")

    # Remove border
    interior = data[border:-border, border:-border]
    print(f"After border removal ({border}px): {interior.shape}")

    if scale:
        # Denormalize: [-1, 1] -> [0, 3] K
        result = (interior + 1) * 1.5
        print(f"After scaling to physical: [{result.min():.4f}, {result.max():.4f}] K")
    else:
        result = interior
        print(f"Interior range: [{np.nanmin(result):.4f}, {np.nanmax(result):.4f}]")

    return result


def calculate_grid_extent(params):
    """Calculate grid origin from domain center coordinates."""
    import pyproj

    # Create the projection
    proj = pyproj.Proj(proj='stere', lat_0=90, lon_0=params['central_longitude'],
                       lat_ts=params['true_scale_latitude'])

    # Convert domain center to projection coordinates
    x_center, y_center = proj(params['domain_center_lon'], params['domain_center_lat'])

    # Calculate grid origin (lower-left corner)
    dx = params['grid_resolution']
    nx = params['nx']
    ny = params['ny']

    x_origin = x_center - (nx / 2) * dx
    y_origin = y_center - (ny / 2) * dx

    return x_origin, y_origin, x_center, y_center


def save_netcdf(data, output_file, params, diagnostic_plot=False):
    """
    Save processed data to netCDF with projection coordinates.

    Parameters
    ----------
    data : array-like
        2D data array (already processed with border removed)
    output_file : str
        Path to save the netCDF file
    params : dict
        Grid parameters dict (determines output resolution)
    diagnostic_plot : bool
        If True, save a diagnostic PNG plot alongside the netCDF
    """
    from scipy.ndimage import zoom

    nx = params['nx']
    ny = params['ny']
    dx = params['grid_resolution']

    print(f"\n=== Saving netCDF ===")
    has_nan_in = np.any(np.isnan(data))
    print(f"Input: shape={data.shape}, min={np.nanmin(data):.4f}, mean={np.nanmean(data):.4f}, max={np.nanmax(data):.4f}, has_nan={has_nan_in}")

    # Resample if data shape doesn't match target grid
    if nx != data.shape[0]:
        scale_factor = nx / data.shape[0]
        data_out = zoom(data, scale_factor, order=1)  # bilinear interpolation
        print(f"Resampled: {data.shape} -> {data_out.shape} (scale={scale_factor:.3f})")
    else:
        data_out = data

    # Diagnostics for output data
    has_nan_out = np.any(np.isnan(data_out))
    print(f"Output: shape={data_out.shape}, min={np.nanmin(data_out):.4f}, mean={np.nanmean(data_out):.4f}, max={np.nanmax(data_out):.4f}, has_nan={has_nan_out}")
    if has_nan_out:
        n_nan = np.sum(np.isnan(data_out))
        print(f"WARNING: {n_nan} NaN values ({100*n_nan/data_out.size:.2f}%)")

    # Optional diagnostic plot
    if diagnostic_plot:
        diag_file = output_file.replace('.nc', '_diagnostic.png')
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(data_out, origin='lower', cmap='jet', vmin=0, vmax=3)
        ax.set_title(f'NetCDF output: {data_out.shape[0]}x{data_out.shape[1]}, dx={dx:.0f}m')
        plt.colorbar(im, ax=ax, label='Uncertainty (K)')
        plt.savefig(diag_file, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Diagnostic plot: {diag_file}")

    x0, y0, _, _ = calculate_grid_extent(params)

    # Create coordinate arrays (cell centers)
    x = x0 + (np.arange(nx) + 0.5) * dx
    y = y0 + (np.arange(ny) + 0.5) * dx

    ds = xr.Dataset(
        data_vars={
            'uncertainty': (['y', 'x'], data_out.astype(np.float32), {
            #'uncertainty': (['y', 'x'], data_out, {
                'units': 'K',
                'long_name': 'Temperature uncertainty estimate',
            })
        },
        coords={
            'x': (['x'], x, {
                'units': 'm',
                'long_name': 'x coordinate in projection',
                'standard_name': 'projection_x_coordinate',
            }),
            'y': (['y'], y, {
                'units': 'm',
                'long_name': 'y coordinate in projection',
                'standard_name': 'projection_y_coordinate',
            }),
        },
        attrs={
            'title': 'CARRA2 uncertainty estimate',
            'projection': 'North Polar Stereographic',
            'central_longitude': params['central_longitude'],
            'domain_center_lon': params['domain_center_lon'],
            'domain_center_lat': params['domain_center_lat'],
            'grid_resolution_m': dx,
        }
    )

    ds.to_netcdf(output_file)
    print(f"Saved netCDF: {output_file}")


def plot_with_coastlines(data, output_file, params, cmap='jet', vmin=None, vmax=None,
                         colorbar=False, gridlines=False):
    """
    Plot data with coastlines using cartopy.

    Parameters
    ----------
    data : array-like
        2D data array (already processed with border removed)
    output_file : str
        Path to save the output figure
    params : dict
        Grid parameters dict containing projection info
    cmap : str
        Colormap name (default: 'jet')
    vmin, vmax : float
        Color scale limits
    colorbar : bool
        Whether to add a colorbar (default: False)
    gridlines : bool
        Whether to add latitude/longitude gridlines with labels (default: False)
    """

    projection = ccrs.NorthPolarStereo(
        central_longitude=params['central_longitude']
    )

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': projection})

    dx = params['grid_resolution']
    nx = params['nx']
    ny = params['ny']

    # Calculate extent from domain center
    x0, y0, x_center, y_center = calculate_grid_extent(params)

    x1 = x0 + nx * dx
    y1 = y0 + ny * dx

    print(f"Grid extent: x=[{x0:.0f}, {x1:.0f}], y=[{y0:.0f}, {y1:.0f}]")
    print(f"Domain center: ({x_center:.0f}, {y_center:.0f})")

    # extent = [left, right, bottom, top]
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   extent=[x0, x1, y0, y1], transform=projection)

    # Add coastlines and features
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # Add gridlines with labels if requested
    if gridlines:
        import matplotlib.ticker as mticker
        gl = ax.gridlines(draw_labels=True, linewidth=1., color='gray',
                          alpha=0.7, linestyle='--',
                          x_inline=False, y_inline=True)
        gl.top_labels = False
        gl.bottom_labels = True
        gl.xlabel_style = {'size': 8, 'color': 'black', 'rotation': 0}
        gl.ylabel_style = {'size': 9, 'color': 'white'}
        gl.xlocator = mticker.FixedLocator(range(-180, 181, 20))
        gl.ylocator = mticker.FixedLocator(range(50, 91, 10))
        #gl.xlabel_style = {'size': 14}
        #gl.ylabel_style = {'size': 14}

    if colorbar:
        plt.colorbar(im, ax=ax, shrink=0.8, label='Kelvin')

    plt.savefig(output_file, bbox_inches='tight', facecolor='white', dpi=100)
    plt.close()
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Add coastlines to CARRA2 uncertainty field plots"
    )
    parser.add_argument(
        "datetime",
        help="DateTime string in format YYYYMMDDHH (e.g., 2019050100)"
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="../sample_data/input",
        help="Input directory containing netCDF files (default: ../sample_data/input)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="../sample_data/output",
        help="Output directory for PNG files (default: ../sample_data/output)"
    )
    parser.add_argument(
        "--colorbar", "-c",
        action="store_true",
        help="Add colorbar to the plot"
    )
    parser.add_argument(
        "--gridlines", "-g",
        action="store_true",
        help="Add lat/lon gridlines with labels"
    )
    parser.add_argument(
        "--vmin",
        type=float, default=0.0,
        help="Minimum value for color scale (default: 0.0)"
    )
    parser.add_argument(
        "--vmax",
        type=float, default=3.0,
        help="Maximum value for color scale (default: 3.0)"
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["netcdf", "netcdf_raw", "png"],
        default="netcdf",
        help="Input mode: netcdf (upscaled), netcdf_raw (256x256), or png (jet colormap)"
    )
    parser.add_argument(
        "--save-netcdf", "-n",
        action="store_true",
        help="Also save processed data as netCDF (physical units, no padding)"
    )
    parser.add_argument(
        "--output-resolution", "-r",
        type=int, choices=[246, 2880],
        default=None,
        help="NetCDF output resolution: 246 (ML model, small), 2880 (full CARRA2, large), or omit for native (754)"
    )

    args = parser.parse_args()


    # Build file paths and read data based on mode
    if args.mode == "png":
        in_file = f"{args.input_dir}/UQ_{args.datetime}.png"
        data = read_png(in_file, vmin=args.vmin, vmax=args.vmax)
        border = 17 # Detected border size for PNG files
        scale = False  # Data is already in physical units after inversion
        params = get_carra2_params(788, 17)         # PNG: 754x754

    elif args.mode == "netcdf_raw":
        in_file = f"{args.input_dir}/UQ_raw_{args.datetime}.nc"
        data = read_netcdf(in_file)
        border = 5  # Border size to remove for raw data
        scale = True  # Need to scale from [-1, 1] to [0, 3] K
        params = get_carra2_params(256, 5)          # Raw netCDF: 246x246

    else:  # netcdf
        in_file = f"{args.input_dir}/UQ_{args.datetime}.nc"
        data = read_netcdf(in_file)
        border = 57  # Border size to remove for upscaled data
        scale = False
        params = get_carra2_params(2880, 57)   # Upscaled netCDF: 2766x2766

    # process data by removing white border and scaling data
    print(f"\nPlotting UQ estimate, shape: {data.shape}")
    data = process_data(data, border=border, scale=scale)

    # Plot with coastlines
    out_file = f"{args.output_dir}/UQ_{args.datetime}.png"
    plot_with_coastlines(
        data, out_file, params,
        vmin=args.vmin, vmax=args.vmax,
        colorbar=args.colorbar,
        gridlines=args.gridlines
    )

    # Save as netCDF if requested
    if args.save_netcdf and args.mode == "png":
        nc_out = out_file.replace('.png', '.nc')
        nc_params = get_carra2_params(args.output_resolution)
        save_netcdf(data, nc_out, nc_params)

    print("\nDone!")
