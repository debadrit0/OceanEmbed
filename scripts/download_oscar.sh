#!/usr/bin/env bash
set -euo pipefail

# NASA PO.DAAC OSCAR Final V2.0, 0.25-degree daily product.
# Requires Earthdata Login credentials configured for podaac-data-subscriber.
# The product page currently lists collection OSCAR_L4_OC_FINAL_V2.0.

START_DATE="${START_DATE:-2020-01-01T00:00:00Z}"
END_DATE="${END_DATE:-2020-12-31T23:59:59Z}"
OUT="${OUT:-data/raw/oscar}"
mkdir -p "$OUT"

podaac-data-downloader \
  -c OSCAR_L4_OC_FINAL_V2.0 \
  -d "$OUT" \
  --start-date "$START_DATE" \
  --end-date "$END_DATE" \
  -b="45,5,105,30" \
  -e .nc

echo "OSCAR files downloaded to $OUT"
