#!/usr/bin/env python3
"""
grib_to_daily_nc.py  (MODIFIED)

Output:
    For each day YYYYMMDD, generate FOUR files:

        CARRA2_SFC_YYYYMMDD_00.nc
        CARRA2_SFC_YYYYMMDD_06.nc
        CARRA2_SFC_YYYYMMDD_12.nc
        CARRA2_SFC_YYYYMMDD_18.nc

Each file contains ALL 4 variables (t2m, sp, u10, v10) for that UTC hour only.
"""

import sys
import os
import re
import numpy as np
import xarray as xr
from collections import defaultdict
from datetime import datetime

try:
    import pygrib
    HAVE_PYGRIB = True
except Exception:
    HAVE_PYGRIB = False


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


SHORTNAME_MAP = {
    "2t": "t2m", "t2m": "t2m",
    "msl": "sp", "sp": "sp",
    "10u": "u10", "u10": "u10",
    "10v": "v10", "v10": "v10"
}

def infer_date_from_filename(fname):
    m = re.search(r"(\d{8})", fname)
    return m.group(1) if m else None


# ---------------------------------------------------------
# READ ONE GRIB FILE INTO perday_store[day][UTC][var]
# ---------------------------------------------------------

def process_file_pygrib(path, store):

    fbase = os.path.basename(path)
    file_date = infer_date_from_filename(fbase)

    try:
        grbs = pygrib.open(path)
    except Exception as e:
        print(f"  ERROR opening {path}: {e}")
        return

    for msg in grbs:
        short = getattr(msg, "shortName", None)
        if not short:
            continue

        short = short.lower()
        varname = SHORTNAME_MAP.get(short)
        if not varname:
            continue

        # values
        try:
            arr = np.array(msg.values, dtype=np.float32)
        except:
            continue

        # grid
        try:
            lats, lons = msg.latlons()
            lat2d = lats.astype(np.float32)
            lon2d = lons.astype(np.float32)
        except:
            lat2d = lon2d = None

        # valid datetime
        valid_dt = getattr(msg, "validDate", None)
        if valid_dt is None and file_date:
            valid_dt = datetime.strptime(file_date, "%Y%m%d")

        if valid_dt is None:
            continue

        daykey = valid_dt.strftime("%Y%m%d")
        hourkey = valid_dt.strftime("%H")   # "00", "06", "12", "18"

        # store grouped by hour
        store[daykey][hourkey][varname].append((valid_dt, arr, lat2d, lon2d))

    grbs.close()


# ---------------------------------------------------------
# WRITE OUTPUT PER DAY + PER UTC
# ---------------------------------------------------------

def write_day_hour(daykey, hourkey, entries, out_dir):

    want = ["t2m", "sp", "u10", "v10"]
    data_vars = {}
    lat2d = lon2d = None

    for v in want:
        if v not in entries:
            continue

        # only ONE record expected per hour
        recs = entries[v]
        if len(recs) == 0:
            continue

        valid_dt, arr, lat2, lon2 = recs[0]

        data_vars[v] = (("y", "x"), arr)

        if lat2 is not None and lon2 is not None:
            lat2d = lat2
            lon2d = lon2

    if not data_vars:
        print(f"  No variables found for {daykey} {hourkey} UTC \u2014 skipping")
        return

    ny, nx = next(iter(data_vars.values()))[1].shape

    coords = {
        "y": ("y", np.arange(ny)),
        "x": ("x", np.arange(nx)),
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    if lat2d is not None:
        ds["lat"] = (("y", "x"), lat2d)
        ds["lon"] = (("y", "x"), lon2d)

    ds = ds.assign_coords(
        valid_datetime=np.array([np.datetime64(valid_dt)], dtype="datetime64[ns]")
    )

    ds.attrs["source"] = "converted by grib_to_daily_nc.py"

    outname = f"CARRA2_SFC_{daykey}_{hourkey}.nc"
    outpath = os.path.join(out_dir, outname)

    print(f"  Writing {outpath}")
    ds.to_netcdf(outpath)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    if len(sys.argv) < 3:
        print("Usage: python grib_to_daily_nc.py input_dir out_dir")
        sys.exit(1)

    input_dir = sys.argv[1]
    out_dir   = sys.argv[2]

    safe_makedirs(out_dir)

    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".grib", ".grb", ".grib2"))
    ])

    if not files:
        print("No GRIB files found")
        sys.exit(1)

    print(f"Processing {len(files)} GRIB files")

    # store[day][hour][var] = list of records
    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for f in files:
        print(f"\nReading {f}")
        if HAVE_PYGRIB:
            process_file_pygrib(f, store)
        else:
            print("ERROR: pygrib required")

    # OUTPUT \u2014 four files per day
    for daykey in sorted(store.keys()):
        print(f"\nAssembling day {daykey}")
        for hourkey in ["00", "06", "12", "18"]:
            write_day_hour(daykey, hourkey, store[daykey][hourkey], out_dir)

    print("\nCompleted.")


if __name__ == "__main__":
    main()
