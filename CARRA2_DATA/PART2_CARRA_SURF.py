import os
import subprocess
from pathlib import Path
import shutil

# Experiment Configuration
EXP = "CARRA2_WINT"
INPUT = f"/scratch/swe4281/DDPM_DATA/{EXP}"
OUTPUT = f"/scratch/swe4281/DDPM_DATA/{EXP}/NCFILES"
os.makedirs(OUTPUT, exist_ok=True)

# Input Date Configuration
DTG = "202201"
DTSTR, DTEND = 20, 21
YY, MM = DTG[:4], DTG[4:6]
print(YY, MM)

# Ensemble Member Configuration
ens_mem = range(10)  # Members from 0 to 9

# Input Files and Parameters
FAFILE = ['ICMSHHARM+0000']

# Load Parameters from External File
params_file = 'list_ALL.txt'
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# Main Processing Loop
for file_name in FAFILE:
    for DD in range(DTSTR, DTEND):  # Adjust this range if needed
        DDN = f"{DD:02}"
        for HH in ['12']:
        #for HH in ['00', '06', '12', '18']:
            for mem in ens_mem:
                mem_IN = Path(f"{INPUT}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}")
                mem_OUT = Path(f"{OUTPUT}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC{file_name[11:14]}")
                EC_DISK = Path(f"ec:/swe4281/DDPM_NCDATA/{EXP}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC{file_name[11:14]}")

                # Create necessary directories
                subprocess.run(['emkdir', '-p', str(EC_DISK)])
                os.makedirs(mem_OUT, exist_ok=True)

                # Change to the input directory
                os.chdir(mem_IN)

                for param in list_params:
                    print(f'Processing {file_name} for parameter {param}')

                    # Run the conversion script
                    command = ['epy_conv.py', '-o', 'nc', '-f', param, str(mem_IN / file_name)]
                    subprocess.run(command)

                    # Move the output .nc file to the output directory
                    src_file = Path(f"{file_name}.nc")
                    dest_file = mem_OUT / f"{param}.nc"
                    if src_file.exists():
                        shutil.move(src_file, dest_file)
                        print(f"Moved {src_file} to {dest_file}")

                    # Copy files to EC disk
                    ec_cp = ['ecp', str(dest_file), str(EC_DISK)]
                    subprocess.run(ec_cp)

                    ec_cp1 = ['ecp', str(mem_IN / file_name), str(EC_DISK)]
                    subprocess.run(ec_cp1)

print("Processing complete.")
