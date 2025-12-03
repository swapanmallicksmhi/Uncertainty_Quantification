# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
import subprocess
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from datetime import datetime
from config import *

def process_dyn(PATH_TO_DATA, VARIABLE, YY, MM, DDN, HH):
    timestamp = f"{YY}{MM}{DDN}{HH}"
    mem_IN = Path(f"{PATH_TO_DATA}")
    ncfile = f"{mem_IN}/CARRA2_SFC_{YY}{MM}{DDN}_{HH}.nc"

    command = ['ln', '-sf', ncfile, "FILE.nc"]
    subprocess.run(command)

    ds = xr.open_dataset("FILE.nc")
    data = xr.open_mfdataset(["FILE.nc"], combine='nested')

    if VARIABLE == "sp":
        data["sp"] = np.exp(data["sp"])

    if VARIABLE in scale_factors:
        data["sp"] *= scale_factors[VARIABLE]

    Dyn = data[VARIABLE]
    ds_processed = xr.Dataset({VARIABLE + "_mean": Dyn})
    ds_processed = ds_processed.expand_dims({'time': [timestamp]})

    min_level, max_level = variable_mean_ranges[VARIABLE]
    levels = np.arange(np.floor(min_level), np.ceil(max_level), 1)

    if "XXX" not in VARIABLE:
        plot_dyn("", Dyn, "hres_scale.png", timestamp, levels)
        plot_dynC("", Dyn, "hres.png", timestamp, levels)

    return ds_processed

# MAIN SCRIPT
params_list = sys.argv[1]
DTG = sys.argv[2]
DTSTR = int(sys.argv[3])
DTEND = int(sys.argv[4])
YY, MM = DTG[:4], DTG[4:6]
INPUT = sys.argv[5]
OUTDIR = sys.argv[6]

SCRPDIR = os.getcwd()
TMPDIR = f"{OUTDIR}/tmp_{params_list}_{DTG}_{DTSTR}_{DTEND}"
INPUT1 = f"{INPUT}"

if ".txt" in params_list:
    with open(params_list, 'r') as f:
        params = [line.strip() for line in f if line.strip()]
else:
    params = [s.strip() for s in params_list.split(',')]

os.makedirs(OUTDIR, exist_ok=True)

for DD in range(DTSTR, DTEND + 1):
    DDN = f"{DD:02}"

    for HH in ['00', '06', '12', '18']:
        timestamp = f"{DTG}{DDN}{HH}"

        for param in params:
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)

            try:
                ds_param = process_dyn(INPUT1, param, YY, MM, DDN, HH)

                OUTPUT = f"{OUTDIR}/{param}"
                os.makedirs(OUTPUT, exist_ok=True)

                for png1 in ['hres_scale', 'hres']:
                    fname = f"{png1}.png"
                    if os.path.exists(fname):
                        outname = f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}_carra2.png"
                        subprocess.run(['mv', fname, outname])

            except Exception as e:
                print(f"For {timestamp}, parameter {param} was not processed: {str(e)}")

            os.chdir(SCRPDIR)
            subprocess.run(["rm", "-rfd", TMPDIR], cwd=SCRPDIR)

quit()
