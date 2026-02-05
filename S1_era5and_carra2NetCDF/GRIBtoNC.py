import xarray as xr

def convert_grib_to_netcdf(input_grib: str, output_netcdf: str):
    """
    Convert a GRIB file to a NetCDF file.
    
    Parameters:
    input_grib (str): Path to the input GRIB file.
    output_netcdf (str): Path to the output NetCDF file.
    """
    try:
        # Open the GRIB file using xarray and cfgrib engine
        ds = xr.open_dataset(input_grib, engine="cfgrib")
        
        # Save the dataset as a NetCDF file
        ds.to_netcdf(output_netcdf)
        
        print(f"Conversion successful: {output_netcdf}")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    # Define input GRIB file and output NetCDF file
    input_grib_file = "IN.grib"  # Change to the actual file path
    output_netcdf_file = "OUT.nc"
    
    # Convert the GRIB file to NetCDF
    convert_grib_to_netcdf(input_grib_file, output_netcdf_file)
