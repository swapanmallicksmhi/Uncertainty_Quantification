#!/bin/bash


# old: PARAM = 131.228/132.228/134/167,
# 131 and 132 is the 10m neutral wind

# 1  day  of ERA5 EDA: 23.6MB GRIB, 46.4MB NetCDF
# 30 days of ERA5 EDA: 708MB  GRIB,  1.4GB NetCDF

module load eclib
module load python3

export MARS_MULTITARGET_STRICT_FORMAT=1
DDIR=/scratch/${HOME}/DDPM_DATA/S1_era5and_carra2NetCDF
SCRDIR=/home/${HOME}/S1_era5and_carra2NetCDF
#SCRDIR=/home/fasg/CARRA2/uncert_est/mars_retrievals

cd $DDIR || { echo "Could cd to $DDIR" ; exit ; }

DATE1=20220301
DATE2=20220330

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
  DATE = ${DATES},
  TIME = 00/06/12/18,
  EXPVER = 1,
  CLASS = EA,
  LEVTYPE = ML,
  LEVELIST= 96/106/119/123,
  GRID = 0.6/0.6,
  NUMBER = 0/1/2/3/4/5/6/7/8/9,
  PARAM = T/U/V/Q,
  PROCESS = LOCAL,
  ROTATION = 0.0/-30.0,
  STREAM = ENDA,
  TYPE = AN,
  AREA = 29.3/-41.1/-38.5/36.3,
  ACCURACY = 16,
  TARGET="$DDIR/ERA5_EDA_ML_[DATE].grib"
RETRIEVE,
  DATE = ${DATES},
  TIME = 00/06/12/18,
  EXPVER = 1,
  CLASS = EA,
  LEVTYPE = SFC,
  GRID = 0.6/0.6,
  NUMBER = 0/1/2/3/4/5/6/7/8/9,
  PARAM = 134/167,
  PROCESS = LOCAL,
  ROTATION = 0.0/-30.0,
  STREAM = ENDA,
  TYPE = AN,
  AREA = 29.3/-41.1/-38.5/36.3,
  ACCURACY = 16,
  TARGET="$DDIR/ERA5_EDA_SFC_[DATE].grib"
EOF

for n in {0..9};do
    cat >> mars_retrieve.dat <<EOF
RETRIEVE,
  DATE = ${DATES},
  TIME = 00/06/12/18,
  EXPVER = 1,
  CLASS = EA,
  LEVTYPE = SFC,
  GRID = 0.6/0.6,
  NUMBER = ${n},
  PARAM = 165.128/166.128,
  PROCESS = LOCAL,
  ROTATION = 0.0/-30.0,
  STREAM = ENDA,
  TYPE = AN,
  AREA = 29.3/-41.1/-38.5/36.3,
  ACCURACY = 16,
  TARGET="$DDIR/ERA5_EDA_SFC_[DATE].grib"
EOF
done
mars < mars_retrieve.dat
#exit

GRIBFILES=$(ls -1 ERA5*grib)
for GRB in $GRIBFILES;do
    NCF=$(echo $GRB | sed s/grib/nc/g)
    echo "$GRB -> $NCF"
    ln -s $GRB IN.grib
    python3 $SCRDIR/GRIBtoNC.py
    /bin/mv OUT.nc $NCF
    /bin/rm IN.grib IN.grib.*.idx
done
