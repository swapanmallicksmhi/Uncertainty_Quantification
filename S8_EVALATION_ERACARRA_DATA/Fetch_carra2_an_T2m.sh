#!/bin/bash
#SBATCH --cpus-per-task=1
#SBATCH --error=prepncdf.log
#SBATCH --output=prepncdf.log
#SBATCH --job-name=prep_ncdf
#SBATCH --ntasks=1
#SBATCH --qos=nf
#SBATCH --time=12:00:00

# January, March, April, June, September, and October 2022
# 202201 202203 202204 202206 202209 202210 

module load eclib
module load python3

export MARS_MULTITARGET_STRICT_FORMAT=1
DDIR=/scratch/swe4281/DDPM_DATA/CARR2DYN/DATA
SCRDIR=/home/swe4281/repository/CARRA_QC2025/Uncertainty_Quantification_git/2019_DATA_ERA_CARRA2

cd $DDIR || { echo "Could cd to $DDIR" ; exit ; }

DATE1=20190101
DATE2=20190102
#DATE2=20201231

# Make list of dates for the MARS retrieve
DATES=""
YMD=$DATE1
while (( "$YMD" <= "$DATE2" ));do
    if [[ $YMD == $DATE1 ]];then
	DATES="$YMD"
    else
	DATES=$DATES"/$YMD"
    fi
    YMD=$(newdate -D $YMD +1)
done

cat > mars_retrieve.dat <<EOF
RETRIEVE,
 class=rr,
 date=${DATES},
 expver=prod,
 levtype=sfc,
 origin=no-ar-pa,
 param=134/165/166/167,
 stream=oper,
 time=00/06/12/18,
 type=an,
 target="$DDIR/CARRA2_SFC_[DATE].grib"
EOF

mars < mars_retrieve.dat
exit

GRIBFILES=$(ls -1 CARRA2*grib)
echo $GRIBFILES
exit 0
for GRB in $GRIBFILES;do
    NCF=$(echo $GRB | sed s/grib/nc/g)
    echo "$GRB -> $NCF"
    ln -s $GRB IN.grib
    python3 $SCRDIR/grib2nc_SURF.py
    /bin/mv OUT.nc $NCF
    /bin/rm IN.grib IN.grib.*.idx
done
