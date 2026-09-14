#!/usr/bin/env bash
set -euo pipefail

# Authenticate first:
# copernicusmarine login
#
# IMPORTANT: verify variable names and dataset IDs with:
# copernicusmarine describe
# before launching long downloads. Product catalogue IDs can evolve.

START_DATE="${START_DATE:-2020-01-01}"
END_DATE="${END_DATE:-2020-12-31}"
XMIN="45"; XMAX="105"; YMIN="5"; YMAX="30"
OUT="data/raw"
mkdir -p "$OUT"

# 1) OSTIA daily SST (problem-statement recommended source)
copernicusmarine subset \
  --dataset-id METOFFICE-GLO-SST-L4-REP-OBS-SST \
  --variables analysed_sst \
  --start-datetime "${START_DATE}T00:00:00" \
  --end-datetime "${END_DATE}T23:59:59" \
  --minimum-longitude "$XMIN" --maximum-longitude "$XMAX" \
  --minimum-latitude "$YMIN" --maximum-latitude "$YMAX" \
  --output-directory "$OUT/ostia" \
  --output-filename ostia.nc

# 2) SSS: daily SMAP/SMOS-derived L4 product
copernicusmarine subset \
  --dataset-id cmems_obs-mob_glo_phy-sss_my_multi_P1D \
  --variables sos \
  --start-datetime "${START_DATE}T00:00:00" \
  --end-datetime "${END_DATE}T23:59:59" \
  --minimum-longitude "$XMIN" --maximum-longitude "$XMAX" \
  --minimum-latitude "$YMIN" --maximum-latitude "$YMAX" \
  --output-directory "$OUT/sss" \
  --output-filename sss.nc

# 3) DUACS daily SLA at 0.25 degree dataset
copernicusmarine subset \
  --dataset-id cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D \
  --variables sla \
  --start-datetime "${START_DATE}T00:00:00" \
  --end-datetime "${END_DATE}T23:59:59" \
  --minimum-longitude "$XMIN" --maximum-longitude "$XMAX" \
  --minimum-latitude "$YMIN" --maximum-latitude "$YMAX" \
  --output-directory "$OUT/ssh" \
  --output-filename ssh.nc

# 4) GLORYS target temperatures. The raw GLORYS grid is finer than the 0.25-deg grid;
#    harmonize.py regrids/interpolates to 0.25 degrees and the 15 standard depths.
copernicusmarine subset \
  --dataset-id cmems_mod_glo_phy_my_0.083deg_P1D-m \
  --variables thetao \
  --start-datetime "${START_DATE}T00:00:00" \
  --end-datetime "${END_DATE}T23:59:59" \
  --minimum-longitude "$XMIN" --maximum-longitude "$XMAX" \
  --minimum-latitude "$YMIN" --maximum-latitude "$YMAX" \
  --minimum-depth 0 --maximum-depth 1000 \
  --output-directory "$OUT/glorys" \
  --output-filename glorys.nc

# Winds: use ASCAT/CCMP as requested by the problem statement. If your chosen
# current Copernicus release is used instead, save U/V under $OUT/wind and pass
# the correct variable names to harmonize.py.

echo "Downloaded core Copernicus subsets. Next: add OSCAR current U/V and wind U/V, then harmonize."
