# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
import shutil
import sys
import xarray as xr
import numpy as np

# Experiment Configuration
EXP = "--????--"
INPUT = f"/scratch/--????--/archive"     # Add Input directory
HPC_OUT = f"/scratch/--????--"           # Add Output directory
TMP_PER = f"/scratch/--????--"           # Add Temporary directory
SCRIPT = f"/scratch/--????--"            # Add Script directory
os.makedirs(TMP_PER, exist_ok=True)

# Input Date Configuration
DTG = "202204"                          # Add YYYYMM
DTSTR, DTEND = 1, 2                     # Add Start_date, End_date
YY, MM = DTG[:4], DTG[4:6]
print(f"Year: {YY}, Month: {MM}")

# Ensemble Members
ens_mem = range(10)  # 0\u20139

# Input file and parameters
FAFILE = 'ICMSHHARM+0006'
params_file = 'list_ALL.txt'

# Load parameter list
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# Main processing
for param in list_params:
    for DD in range(DTSTR, DTEND):
        DDN = f"{DD:02}"
        #for HH in ['00','06','12','18']:  # Can expand this list
        for HH in ['00']:  # Can expand this list
            mem_OUT = Path(f"{TMP_PER}/{param}")
            HPC_OUT1 = Path(f"{HPC_OUT}/{param}")
            TMP = Path(f"{TMP_PER}/{param}/tmp")
            os.makedirs(mem_OUT, exist_ok=True)
            os.makedirs(HPC_OUT1, exist_ok=True)
            os.makedirs(TMP, exist_ok=True)

            # Process each ensemble member
            for mem in ens_mem:
                mem_IN = Path(f"{INPUT}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}")
                print(f'Processing {FAFILE} for parameter {param} in {mem_IN}')
                if not mem_IN.exists():
                    print(f"Warning: Input path {mem_IN} does not exist. Skipping.")
                #os.chdir(mem_IN)
                os.chdir(TMP)
                command1 = ['cp', '-f', f"{mem_IN}/{FAFILE}", 'TM_FC']
                subprocess.run(command1, check=True)
                #
                command2 = ['epy_conv.py', '-o', 'nc', '-f', f"{param}", 'TM_FC']
                print(command2)
                subprocess.run(command2, check=True)
                os.remove('TM_FC') 
                print(TMP)

                src_file  = Path(f"TM_FC.nc")
                dest_file = Path(f"{mem}.nc")
                if src_file.exists():
                    shutil.move(src_file, dest_file)
                    continue
                else:
                    print(f"Warning: Expected file {src_file} not found.")

            # Combine files into single NetCDF
            os.chdir(TMP)
            print(TMP)
            file_names = [f"{i}.nc" for i in ens_mem]
            existing_files = [f for f in file_names if Path(f).exists()]

            if not existing_files:
                print(f"No files found to combine for {param} {YY}{MM}{DDN}{HH}.")
                continue

            try:
                ds_all = xr.open_mfdataset(existing_files, combine='nested', concat_dim='ensemble')
                if param in ds_all:
                    data = ds_all[param].expand_dims(dim={'variable': [param]})
                    combined = xr.concat([data], dim='variable')
                    output_file = HPC_OUT1 / f"{param}_{YY}{MM}{DDN}{HH}.nc"
                    combined.to_netcdf(output_file)
                    print(f"NetCDF saved: {output_file}")
                else:
                    print(f"Variable '{param}' not found in dataset for {YY}{MM}{DDN}{HH}.")
            except Exception as e:
                print(f"Error while processing {param}: {e}")
            shutil.rmtree(TMP)
            # Ensemble Created

print("All processing complete.")
