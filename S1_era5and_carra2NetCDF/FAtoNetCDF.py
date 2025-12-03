# -*- coding: utf-8 -*-
#--  Swapan Mallick
#--  8 May 2025
#-
import os
import subprocess
from pathlib import Path
import shutil

# Experiment Configuration
SCRIPT=f"/home/swe4281/POSTPROCESS_EPS/DL_UNCERTAINTY/CARRA2_ENS_JUNE2025"
INPUT = f"/ec/res4/scratch/nhz/hm_home/carra2_str2br_202203/archive"
OUTPUT = f"/scratch/swe4281/DDPM_DATA/KESP/NCFILES"
os.makedirs(OUTPUT, exist_ok=True)

# Input Date Configuration
DTG = "202201"
DTSTR, DTEND = 5, 32
YY, MM = DTG[:4], DTG[4:6]
print(YY, MM)

# Ensemble Member Configuration
ens_mem = range(10)  # Members from 0 to 9

# Input Files and Parameters
FAFILE = ['ICMSHHARM+0006']

# Load Parameters from External File
params_file = 'list_params_carra2.txt'
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{FAFILE}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# Main Processing Loop
#for file_name in FAFILE:
FAFILE = 'ICMSHHARM+0006'
for param in list_params:
    for DD in range(DTSTR, DTEND):  # Adjust this range if needed
        DDN = f"{DD:02}"
        for HH in ['00', '06', '12', '18']:
            TMP = Path(f"/scratch/swe4281/DDPM_DATA/tmp001")
            os.makedirs(TMP, exist_ok=True)
            shutil.rmtree(TMP)
            os.makedirs(TMP, exist_ok=True)
            os.chdir(TMP)
            print('SWAPAN',TMP)
            for mem in ens_mem:
                mem_IN = Path(f"{INPUT}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}")
                mem_OUT = Path(f"{OUTPUT}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC{FAFILE[11:14]}")
                os.makedirs(mem_OUT, exist_ok=True)
                #
                print(f'Processing for parameter {param}')
                command1 = ['ln', '-sf', f"{mem_IN}/{FAFILE}", 'TM_FC']
                subprocess.run(command1, check=True)
                # Run the conversion script
                command2 = ['epy_conv.py', '-o', 'nc', '-f', f"{param}", 'TM_FC']
                subprocess.run(command2,check=True)
                os.remove('TM_FC')
                # Move the output .nc file to the output directory
                src_file  = Path(f"TM_FC.nc")
                dest_file = mem_OUT / f"{param}.nc"
                if src_file.exists():
                    shutil.move(src_file, dest_file)
                    print(f"Moved {src_file} to {dest_file}")
            # ens mem
            os.chdir(SCRIPT)
            shutil.rmtree(TMP)
        # HOUR

print("Processing complete.")
