#!/bin/sh
#-------------------------------
##SBATCH --job-name=coastelline
##SBATCH --output=coastel.out
##SBATCH --error=coastel.out
##SBATCH --nodes=1
##SBATCH --mem=60GB
##SBATCH --time=36:00:00
#-------------------------------

# =============================================================================
# Run script for adding coastlines to CARRA2 uncertainty plots
# Full year 2019, 4 UTC times per day
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ENV_NAME="climate"
INPUT_DIR="XXX"
OUTPUT_DIR="YYY"

MODE="png"

SHOW_COLORBAR=1
SHOW_GRIDLINES=1
SAVE_NETCDF=1

OUTPUT_RESOLUTION="2880"
#OUTPUT_RESOLUTION="2880"

VMIN=0.0
VMAX=3.0

START_DATE="2019-01-01"
END_DATE="2019-12-31"
UTC_HOURS=("00" "06" "12" "18")

# -----------------------------------------------------------------------------
# Activate conda environment
# -----------------------------------------------------------------------------
echo "Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

if [ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: Failed to activate conda environment '$ENV_NAME'"
    exit 1
fi

echo "Environment activated successfully"
echo "Python: $(which python)"
echo ""

# -----------------------------------------------------------------------------
# Verify input directory and create output directory
# -----------------------------------------------------------------------------
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Build command-line options
# -----------------------------------------------------------------------------
OPTS="--mode $MODE"
[ "$SHOW_COLORBAR" -eq 1 ] && OPTS="$OPTS --colorbar"
[ "$SHOW_GRIDLINES" -eq 1 ] && OPTS="$OPTS --gridlines"
[ "$SAVE_NETCDF" -eq 1 ] && OPTS="$OPTS --save-netcdf"
[ -n "$OUTPUT_RESOLUTION" ] && OPTS="$OPTS --output-resolution $OUTPUT_RESOLUTION"

# -----------------------------------------------------------------------------
# Main loop: full year 2019, 4 UTC per day
# -----------------------------------------------------------------------------
echo "Processing full year 2019 (4 UTC per day)..."
echo ""

current_date="$START_DATE"

while [ "$current_date" != "$(date -I -d "$END_DATE + 1 day")" ]; do
    ymd=$(date -d "$current_date" +"%Y%m%d")

    for hh in "${UTC_HOURS[@]}"; do
        DT="${ymd}${hh}"

        echo "=== Processing: $DT ==="
        python add_coastlines.py "$DT" \
            --input-dir "$INPUT_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --vmin "$VMIN" \
            --vmax "$VMAX" \
            $OPTS
        echo ""
    done

    current_date=$(date -I -d "$current_date + 1 day")
done

echo "All done!"
