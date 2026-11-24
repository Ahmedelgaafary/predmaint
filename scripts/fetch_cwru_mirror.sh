#!/bin/sh
# Download the standard CWRU subset (12 kHz drive end: 3 fault sizes x 3 fault types x 4 loads, plus 4 normal
# baselines = 40 files, ~135 MB) from a community GitHub mirror, using a sparse git checkout.
#
# The authoritative source is the CWRU Bearing Data Center (engineering.case.edu/bearingdatacenter). This
# mirror was used to validate the loader; if you can, prefer the official files. Layout does not matter to
# predmaint: files are matched by their leading file number.
#
# usage: sh scripts/fetch_cwru_mirror.sh [target_dir]      (default: ./data/cwru)
set -e
DEST="${1:-data/cwru}"
rm -rf "$DEST" && mkdir -p "$(dirname "$DEST")"
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/s-whynot/CWRU-dataset.git "$DEST"
cd "$DEST"
git sparse-checkout init --no-cone
printf '%s\n' '/Normal/' \
  '/12k_Drive_End_Bearing_Fault_Data/B/007/' '/12k_Drive_End_Bearing_Fault_Data/B/014/' '/12k_Drive_End_Bearing_Fault_Data/B/021/' \
  '/12k_Drive_End_Bearing_Fault_Data/IR/007/' '/12k_Drive_End_Bearing_Fault_Data/IR/014/' '/12k_Drive_End_Bearing_Fault_Data/IR/021/' \
  '/12k_Drive_End_Bearing_Fault_Data/OR/007/@6/' '/12k_Drive_End_Bearing_Fault_Data/OR/014/' '/12k_Drive_End_Bearing_Fault_Data/OR/021/@6/' \
  > .git/info/sparse-checkout
git checkout
echo "files: $(find . -name '*.mat' | wc -l) (expected 40)"
