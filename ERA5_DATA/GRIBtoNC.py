#---Swapan Mallick
#---1 January 2025
import xarray as xr

def convert_grib_to_netcdf(input_grib: str, output_netcdf: str):
    try:
        ds = xr.open_dataset(input_grib, engine="cfgrib")
        ds.to_netcdf(output_netcdf)
        print(f"Conversion successful: {output_netcdf}")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    # Define input GRIB file and output NetCDF file
    input_grib_file = "IN.grib"  # Change to the actual file path
    output_netcdf_file = "OUT.nc"
    convert_grib_to_netcdf(input_grib_file, output_netcdf_file)
