#!/usr/bin/env bash
#SBATCH --output=log.out
#SBATCH --error=log.out
#SBATCH --job-name=CONV2ZARR
#SBATCH --qos=nf
module load python3
python3 CONVERT2ZARR.py
