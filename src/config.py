from pathlib import Path
import numpy as np

# Domain configuration: North Indian Ocean domain
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
GRID_STEP = 0.25

LATS = np.round(np.arange(LAT_MIN, LAT_MAX + GRID_STEP/2, GRID_STEP), 5)
LONS = np.round(np.arange(LON_MIN, LON_MAX + GRID_STEP/2, GRID_STEP), 5)

# Standard 15 target ocean depth levels (0m - 1000m)
DEPTHS_M = np.array([
    0, 5, 10, 20, 30,
    50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000
], dtype=np.float32)

FEATURES = [
    "sst", "sss", "ssh",
    "u_current", "v_current",
    "u_wind", "v_wind"
]

N_INPUTS = len(FEATURES)
N_DEPTHS = len(DEPTHS_M)

PATCH_SIZE = 64
PATCH_STRIDE = 48
RANDOM_PATCHES_PER_DAY = 16

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
for p in (RAW, PROCESSED, ARTIFACTS):
    p.mkdir(parents=True, exist_ok=True)

print(f"Domain: {LAT_MIN}-{LAT_MAX}N, {LON_MIN}-{LON_MAX}E")
print(f"Grid: {len(LATS)} lat x {len(LONS)} lon = {len(LATS)*len(LONS):,} cells/day")
print(f"Depths: {DEPTHS_M.tolist()}")
