#!/usr/bin/env python3
"""
OceanEmbed Data Acquisition Tool
--------------------------------
Downloads operational satellite surface observations and subsurface reanalysis data
from Copernicus Marine Service (CMEMS) for the North Indian Ocean domain (5°N-30°N, 45°E-105°E).

Prerequisites for live Copernicus downloads:
    1. Register for a free Copernicus Marine account at https://marine.copernicus.eu
    2. Run: copernicusmarine login (or set COPERNICUSMARINE_CACHE_DIR and credentials)

Usage:
    # 1. Download real sample data (3 days):
    python download_data.py --sample

    # 2. Download full custom date range:
    python download_data.py --start-date 2023-01-01 --end-date 2023-01-31

    # 3. Generate synthetic sample data (no Copernicus login needed):
    python download_data.py --synthetic
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

# Domain boundaries (North Indian Ocean)
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Copernicus Marine product configurations
PRODUCTS = {
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],
        "filename": "sst.nc",
        "desc": "OSTIA Level-4 Sea Surface Temperature (0.05° regridded to 0.25°)"
    },
    "sss": {
        "dataset_id": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
        "variables": ["sos"],
        "filename": "sss.nc",
        "desc": "SMOS/SMAP Multi-Mission Sea Surface Salinity"
    },
    "ssh": {
        "dataset_id": "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
        "variables": ["sla"],
        "filename": "ssh.nc",
        "desc": "DUACS Multi-Mission Sea Level Anomaly (0.25°)"
    },
    "current": {
        "dataset_id": "cmems_obs-mob_glo_phy-cur_my_0.25deg_P1D-m",
        "variables": ["uo", "vo"],
        "filename": "current.nc",
        "desc": "Total Surface Currents (Geostrophic + Ekman)"
    },
    "wind_asc": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l3-metopb-ascat-asc-0.25deg_P1D-i",
        "variables": ["eastward_wind", "northward_wind"],
        "filename": "wind_asc.nc",
        "desc": "ASCAT MetOp-B Ocean Surface 10m Wind (Ascending pass)"
    },
    "wind_des": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l3-metopb-ascat-des-0.25deg_P1D-i",
        "variables": ["eastward_wind", "northward_wind"],
        "filename": "wind_des.nc",
        "desc": "ASCAT MetOp-B Ocean Surface 10m Wind (Descending pass)"
    },
    "glorys_target": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["thetao"],
        "filename": "glorys_subsurface.nc",
        "desc": "GLORYS12V1 Subsurface In-situ Temperature Reanalysis (0m - 1000m)"
    }
}


def download_copernicus_subset(dataset_id, variables, start_date, end_date, output_dir, filename, min_depth=None, max_depth=None):
    """Download spatio-temporal subset via copernicusmarine Python client."""
    try:
        import copernicusmarine
    except ImportError:
        print("[ERROR] copernicusmarine package is required. Install via: pip install copernicusmarine")
        sys.exit(1)

    out_file = Path(output_dir) / filename
    print(f"\n--> Fetching {filename} ({', '.join(variables)})...")
    print(f"    Dataset: {dataset_id}")
    print(f"    Period:  {start_date} to {end_date}")
    print(f"    Domain:  {LAT_MIN}°N-{LAT_MAX}°N, {LON_MIN}°E-{LON_MAX}°E")

    subset_kwargs = {
        "dataset_id": dataset_id,
        "variables": variables,
        "start_datetime": f"{start_date}T00:00:00",
        "end_datetime": f"{end_date}T23:59:59",
        "minimum_longitude": LON_MIN,
        "maximum_longitude": LON_MAX,
        "minimum_latitude": LAT_MIN,
        "maximum_latitude": LAT_MAX,
        "output_directory": str(output_dir),
        "output_filename": filename,
        "overwrite": True
    }
    if min_depth is not None:
        subset_kwargs["minimum_depth"] = min_depth
    if max_depth is not None:
        subset_kwargs["maximum_depth"] = max_depth

    try:
        copernicusmarine.subset(**subset_kwargs)
        print(f"    [SUCCESS] Saved {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as exc:
        print(f"    [FAILED] {exc}")
        print("    If authentication failed, please run: copernicusmarine login")
        return False


def generate_synthetic_data(output_dir, n_days=5):
    """
    Generate realistic synthetic ocean datasets for immediate local development,
    pipeline testing, and model training without needing Copernicus credentials.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 65)
    print("GENERATING SYNTHETIC OBSERVATIONAL DATASETS")
    print(f"Destination: {out_path.resolve()}")
    print("=" * 65)

    lats = np.arange(LAT_MIN, LAT_MAX + 0.001, 0.25, dtype=np.float32)
    lons = np.arange(LON_MIN, LON_MAX + 0.001, 0.25, dtype=np.float32)
    n_lat = len(lats)
    n_lon = len(lons)
    depths = np.array(DEPTHS, dtype=np.float32)

    try:
        import xarray as xr
        import pandas as pd

        dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
        lon_2d, lat_2d = np.meshgrid(lons, lats)

        # 1. SST (Slight latitudinal gradient + seasonal noise)
        base_sst = 29.5 - 0.25 * (lat_2d - 5.0) + 0.4 * np.sin(np.deg2rad(lon_2d))
        sst_data = np.stack([base_sst + np.random.normal(0, 0.2, (n_lat, n_lon)) + 273.15 for _ in range(n_days)])
        ds_sst = xr.Dataset(
            {"analysed_sst": (["time", "latitude", "longitude"], sst_data.astype(np.float32))},
            coords={"time": dates, "latitude": lats, "longitude": lons}
        )
        ds_sst.to_netcdf(out_path / "sst.nc")
        print("  [✓] Created sst.nc (OSTIA simulated)")

        # 2. SSS (Bay of Bengal fresher ~32 PSU, Arabian Sea saline ~36 PSU)
        base_sss = 34.0 + 1.8 * np.tanh((65.0 - lon_2d) / 10.0)
        sss_data = np.stack([base_sss + np.random.normal(0, 0.1, (n_lat, n_lon)) for _ in range(n_days)])
        ds_sss = xr.Dataset(
            {"sos": (["time", "latitude", "longitude"], sss_data.astype(np.float32))},
            coords={"time": dates, "latitude": lats, "longitude": lons}
        )
        ds_sss.to_netcdf(out_path / "sss.nc")
        print("  [✓] Created sss.nc (SMOS/SMAP simulated)")

        # 3. SSH / SLA (Mesoscale eddies -0.15m to +0.15m)
        ssh_data = np.stack([
            0.12 * np.sin(lon_2d / 4.0) * np.cos(lat_2d / 3.0) + np.random.normal(0, 0.02, (n_lat, n_lon))
            for _ in range(n_days)
        ])
        ds_ssh = xr.Dataset(
            {"sla": (["time", "latitude", "longitude"], ssh_data.astype(np.float32))},
            coords={"time": dates, "latitude": lats, "longitude": lons}
        )
        ds_ssh.to_netcdf(out_path / "ssh.nc")
        print("  [✓] Created ssh.nc (DUACS SLA simulated)")

        # 4. Surface Currents (uo, vo)
        u_cur = np.stack([0.25 * np.cos(lat_2d / 5.0) + np.random.normal(0, 0.05, (n_lat, n_lon)) for _ in range(n_days)])
        v_cur = np.stack([0.15 * np.sin(lon_2d / 6.0) + np.random.normal(0, 0.05, (n_lat, n_lon)) for _ in range(n_days)])
        ds_cur = xr.Dataset(
            {"uo": (["time", "latitude", "longitude"], u_cur.astype(np.float32)),
             "vo": (["time", "latitude", "longitude"], v_cur.astype(np.float32))},
            coords={"time": dates, "latitude": lats, "longitude": lons}
        )
        ds_cur.to_netcdf(out_path / "current.nc")
        print("  [✓] Created current.nc (OSCAR/Total currents simulated)")

        # 5. ASCAT Winds
        u_wind = np.stack([4.5 + 2.0 * np.sin(lat_2d / 4.0) + np.random.normal(0, 0.5, (n_lat, n_lon)) for _ in range(n_days)])
        v_wind = np.stack([3.0 + 1.5 * np.cos(lon_2d / 5.0) + np.random.normal(0, 0.5, (n_lat, n_lon)) for _ in range(n_days)])
        ds_wind = xr.Dataset(
            {"eastward_wind": (["time", "latitude", "longitude"], u_wind.astype(np.float32)),
             "northward_wind": (["time", "latitude", "longitude"], v_wind.astype(np.float32))},
            coords={"time": dates, "latitude": lats, "longitude": lons}
        )
        ds_wind.to_netcdf(out_path / "wind_asc.nc")
        ds_wind.to_netcdf(out_path / "wind_des.nc")
        print("  [✓] Created wind_asc.nc & wind_des.nc (ASCAT winds simulated)")

        # 6. GLORYS Subsurface Temperature Targets (15 depths)
        target_temp = np.zeros((n_days, len(depths), n_lat, n_lon), dtype=np.float32)
        for d_idx, d_m in enumerate(depths):
            thermocline_factor = np.exp(-d_m / 220.0)
            target_temp[:, d_idx, :, :] = (
                7.5 + (base_sst - 7.5) * thermocline_factor
                + np.random.normal(0, 0.25, (n_days, n_lat, n_lon))
            )

        ds_target = xr.Dataset(
            {"thetao": (["time", "depth", "latitude", "longitude"], target_temp)},
            coords={"time": dates, "depth": depths, "latitude": lats, "longitude": lons}
        )
        ds_target.to_netcdf(out_path / "glorys_subsurface.nc")
        print("  [✓] Created glorys_subsurface.nc (15 depths target reanalysis)")

    except ImportError:
        print("  [INFO] xarray or netCDF4 not installed. Generating raw NumPy arrays (.npz format)...")
        np.savez_compressed(
            out_path / "raw_simulated.npz",
            lats=lats,
            lons=lons,
            depths=depths,
            sst=28.5 + np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            sss=35.0 + np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            ssh=0.05 * np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            u_cur=0.2 * np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            v_cur=0.2 * np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            u_wind=5.0 + np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            v_wind=3.0 + np.random.randn(n_days, n_lat, n_lon).astype(np.float32),
            thetao=20.0 + np.random.randn(n_days, len(depths), n_lat, n_lon).astype(np.float32)
        )
        print("  [✓] Created raw_simulated.npz")

    print("\n[SUCCESS] Synthetic raw dataset successfully generated in", out_path.resolve())
    print("Next step: run 'python create_dataset.py' to process into training patches.")


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed Data Acquisition & Download Utility")
    parser.add_argument("--start-date", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2023-01-03", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Directory to save downloaded files")
    parser.add_argument("--sample", action="store_true", help="Download 3-day sample (2023-01-01 to 2023-01-03)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic sample data locally without Copernicus login")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        generate_synthetic_data(out_dir)
        return

    start_date = "2023-01-01" if args.sample else args.start_date
    end_date = "2023-01-03" if args.sample else args.end_date

    print("=" * 65)
    print("OCEANEMBED COPERNICUS MARINE DOWNLOADER")
    print(f"Domain:      {LAT_MIN}°N - {LAT_MAX}°N, {LON_MIN}°E - {LON_MAX}°E")
    print(f"Date Range:  {start_date} to {end_date}")
    print(f"Output:      {out_dir.resolve()}")
    print("=" * 65)

    failed = 0
    for key, info in PRODUCTS.items():
        min_d = 0 if key == "glorys_target" else None
        max_d = 1000 if key == "glorys_target" else None
        success = download_copernicus_subset(
            dataset_id=info["dataset_id"],
            variables=info["variables"],
            start_date=start_date,
            end_date=end_date,
            output_dir=out_dir,
            filename=info["filename"],
            min_depth=min_d,
            max_depth=max_d
        )
        if not success:
            failed += 1

    if failed > 0:
        print("\n" + "!" * 65)
        print(f"[NOTE] {failed} downloads were not completed.")
        print("Tip: You can generate synthetic sample data without credentials using:")
        print("     python download_data.py --synthetic")
        print("!" * 65)


if __name__ == "__main__":
    main()
