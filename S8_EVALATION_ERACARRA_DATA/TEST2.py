#!/usr/bin/env python3
"""
Create ONE NetCDF per day.

Input directory contains files such as:
    CARRA2_SFC_20190101_00.grib
    CARRA2_SFC_20190101_06.grib
    CARRA2_SFC_20190101_12.grib
    CARRA2_SFC_20190101_18.grib

For each day, extract variables t2m, sp, u10, v10 and save all into:
    YYYYMMDD.nc
"""

import sys
import os
import re
import numpy as np
import xarray as xr
from datetime import datetime
from collections import defaultdict

# pygrib backend (preferred)
try:
    import pygrib
    HAVE_PYGRIB = True
except:
    HAVE_PYGRIB = False


# ---------------------------------------------------
# Helper
# ---------------------------------------------------
def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def extract_date_and_hour(filename):
    """
    Extract YYYYMMDD and HH from filenames like:
      CARRA2_SFC_20190101_06.grib
      CARRA2_SFC_20190101.grib  \u2192 assume hour=00
    """
    base = os.path.basename(filename)

    m = re.search(r"(\d{8})_(\d{2})", base)
    if m:
        return m.group(1), m.group(2)

    # fallback (no hour present \u2192 assume 00)
    m2 = re.search(r"(\d{8})", base)
    if m2:
        return m2.group(1), "00"

    return None, None


# ---------------------------------------------------
# Extract a single variable from one GRIB file
# ---------------------------------------------------
def read_variable_pygrib(grib_file, shortname):
    grbs = pygrib.open(grib_file)
    arr = None
    valid_dt = None
    lat2d = lon2d = None

    for msg in grbs:
        if msg.shortName == shortname:
            arr = msg.values.astype(np.float32)
            try:
                lats, lons = msg.latlons()
                lat2d = lats.astype(np.float32)
                lon2d = lons.astype(np.float32)
            except:
                pass
            valid_dt = getattr(msg, "validDate", None)
            break

    grbs.close()
    return arr, lat2d, lon2d, valid_dt


# ---------------------------------------------------
# MAIN DAILY PROCESSING
# ---------------------------------------------------
def process_daily_group(day, files, out_dir):
    print(f"\n=== Processing day {day} ({len(files)} files) ===")

    wanted_vars = ["t2m", "sp", "u10", "v10"]
    data_dict = {var: [] for var in wanted_vars}
    time_list = []
    lat2d = lon2d = None

    for f in sorted(files):
        date, hour = extract_date_and_hour(f)
        time_list.append(np.datetime64(f"{date}T{hour}:00"))

        for var in wanted_vars:
            arr, lat_tmp, lon_tmp, _ = read_variable_pygrib(f, var)

            if arr is None:
                print(f"  Warning: variable {var} missing in {f}")
                continue

            if lat2d is None and lat_tmp is not None:
                lat2d, lon2d = lat_tmp, lon_tmp

            data_dict[var].append(arr)

    # Convert to stacks
    ds = xr.Dataset()

    for var in wanted_vars:
        arr_list = data_dict[var]
        if len(arr_list) == 0:
            print(f"  Variable {var} missing for day {day}, skipping.")
            continue

        arr_stack = np.stack(arr_list, axis=0)  # (time, y, x)
        ds[var] = (("time", "y", "x"), arr_stack)

    # Coordinates
    ds = ds.assign_coords(time=("time", np.array(time_list)))
    ny, nx = arr_stack.shape[1:]
    ds = ds.assign_coords(y=np.arange(ny), x=np.arange(nx))

    if lat2d is not None:
        ds["lat"] = (("y", "x"), lat2d)
        ds["lon"] = (("y", "x"), lon2d)

    # Save daily file
    outfile = os.path.join(out_dir, f"{day}.nc")
    print(f"  \u2192 Writing {outfile}")
    ds.to_netcdf(outfile)
    print("  Done.\n")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python grib_daily_to_netcdf.py input_dir out_dir")
        sys.exit(1)

    input_dir = sys.argv[1]
    out_dir = sys.argv[2]
    safe_makedirs(out_dir)

    # Collect all GRIB files
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".grib", ".grb", ".grib2"))
    ]

    if not files:
        print("No GRIB files found.")
        sys.exit(1)

    print(f"Found {len(files)} GRIB files.\n")

    # Group files by day
    daily_groups = defaultdict(list)

    for f in files:
        day, hour = extract_date_and_hour(f)
        if day:
            daily_groups[day].append(f)

    # Process each day separately
    for day, flist in sorted(daily_groups.items()):
        process_daily_group(day, flist, out_dir)

    print("\nAll days processed. Finished.")


if __name__ == "__main__":
    main()
