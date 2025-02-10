import os
import subprocess
from pathlib import Path
import shutil

# Define variables
EXP1    = "CARRA2"
# Modify the input directory--

SCRPDIR = "/home/XXX"
TMPDIR = f"{SCRPDIR}/tmp_ERA_C"
INPUT1 = f"/scratch/AAA/{EXP1}/NCFILES"
PLOT_OUT = f"{SCRPDIR}/PLOTS"

#
# Input Date Configuration
DTG = "202201"
DTSTR, DTEND = 20, 21
YY, MM = DTG[:4], DTG[4:6]
#
os.makedirs(PLOT_OUT, exist_ok=True)
VARIABLES_PNG = ['SD', 'EnsMEAN']
#
print(YY, MM)

# Ensemble Member Configuration
ens_mem = range(10)  # Members from 0 to 9
# Input Files and Parameters

# Load Parameters from External File
params_file = 'list_params_UV.txt'
if not os.path.exists(params_file):
    raise FileNotFoundError(f"Parameter file '{params_file}' not found.")
with open(params_file, 'r') as f:
    list_params = [line.strip() for line in f if line.strip()]

# Main Processing Loop
#OUTPUT=f"{PLOT_OUT}/{DTG}" 
#os.makedirs(OUTPUT, exist_ok=True)
for DD in range(DTSTR, DTEND):  # Adjust this range if needed
    DDN = f"{DD:02}"
    for HH in ['00', '06', '12', '18']:
        for param in list_params:
            #
            os.makedirs(TMPDIR, exist_ok=True)
            os.chdir(TMPDIR)
            print(f"Image {param} generated")
            #  
            for mem in ens_mem:
                 mem_IN = Path(f"{INPUT1}/{YY}/{MM}/{DDN}/{HH}/mbr00{mem}/FC000")     # CARRA2ENDA
                 print(mem_IN)
                 print(f'Processing Member {mem_IN}')
                 command = ['ln', '-sf', f"{mem_IN}/{param}.nc", f"FILE{mem}.nc"]
                 subprocess.run(command)
                 #
            # create figure---
            image_gen = ['python3', f"{SCRPDIR}/SUB_FINAL_ALL_UV.py", f"{param}"]
            subprocess.run(image_gen)
            #quit()
            # mv figure---
            OUTPUT=f"{PLOT_OUT}/{DTG}/{param}"
            os.makedirs(OUTPUT, exist_ok=True)
            for png1 in VARIABLES_PNG:
                file_out=f"{OUTPUT}/{png1}_{param}_{YY}{MM}{DDN}{HH}.png"
                mvfile = ['mv', f"{png1}.png", f"{file_out}"]
                subprocess.run(mvfile)
            # delete TMPDIR---
            os.chdir(SCRPDIR)
            subprocess.run(["rm", "-rf", TMPDIR], cwd=SCRPDIR)
    #  HH
quit()
