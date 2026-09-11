import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import argopy


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# OCEANEMBED IMPORTS
# ============================================================

from config import LATS, LONS
from model_embed import OceanEmbedEmbeddingModel
from online_year_data import load_surface_block, normalize_surface


# ============================================================
# CONFIGURATION
# ============================================================

YEAR = 2024

# Pilot ARGO region.
ARGO_LON_MIN = 80.0
ARGO_LON_MAX = 90.0
ARGO_LAT_MIN = 10.0
ARGO_LAT_MAX = 20.0

ARGO_START = "2024-08-01"
ARGO_END = "2024-09-01"

# First pilot: only 10 profiles.
MAX_PROFILES = 10

# Same 15 depths used by OceanEmbed.
DEPTHS = np.array(
    [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ],
    dtype=np.float32,
)

# OceanEmbed V2 checkpoint.
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "oceanembed_embedding_v2_2023.pt",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

CSV_OUTPUT = os.path.join(
    RESULTS_DIR,
    "validation_argo_pilot_2024.csv",
)

SUMMARY_OUTPUT = os.path.join(
    RESULTS_DIR,
    "validation_argo_pilot_2024_summary.txt",
)

DEVICE = torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_oceanembed_model():
    print()
    print("=" * 70)
    print("LOADING OCEANEMBED V2")
    print("=" * 70)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            "OceanEmbed V2 checkpoint not found:\n"
            + CHECKPOINT_PATH
        )

    model = OceanEmbedEmbeddingModel().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    target_mean = np.asarray(
        checkpoint["target_mean"],
        dtype=np.float32,
    )

    target_std = np.asarray(
        checkpoint["target_std"],
        dtype=np.float32,
    )

    print(
        "Checkpoint :",
        CHECKPOINT_PATH,
    )

    print(
        "Device     :",
        DEVICE,
    )

    print(
        "Embedding  : 64"
    )

    print(
        "Depths     :",
        len(DEPTHS),
    )

    return (
        model,
        target_mean,
        target_std,
    )


# ============================================================
# TIME CONVERSION
# ============================================================

def convert_time_value(
    value,
    time_variable,
):
    """
    Convert ARGO TIME into Python datetime.
    """

    if isinstance(
        value,
        np.datetime64,
    ):
        return pd.Timestamp(
            value
        ).to_pydatetime()

    if isinstance(
        value,
        datetime,
    ):
        return value

    if hasattr(
        value,
        "year",
    ) and hasattr(
        value,
        "month",
    ):
        return datetime(
            int(value.year),
            int(value.month),
            int(value.day),
            int(
                getattr(
                    value,
                    "hour",
                    0,
                )
            ),
            int(
                getattr(
                    value,
                    "minute",
                    0,
                )
            ),
            int(
                getattr(
                    value,
                    "second",
                    0,
                )
            ),
        )

    try:
        from netCDF4 import num2date

        converted = num2date(
            value,
            units=time_variable.units,
            calendar=getattr(
                time_variable,
                "calendar",
                "standard",
            ),
        )

        if hasattr(
            converted,
            "year",
        ):
            return datetime(
                int(converted.year),
                int(converted.month),
                int(converted.day),
                int(
                    getattr(
                        converted,
                        "hour",
                        0,
                    )
                ),
                int(
                    getattr(
                        converted,
                        "minute",
                        0,
                    )
                ),
                int(
                    getattr(
                        converted,
                        "second",
                        0,
                    )
                ),
            )

    except Exception:
        pass

    return pd.Timestamp(
        value
    ).to_pydatetime()


# ============================================================
# FETCH ARGO
# ============================================================

def fetch_argo():
    print()
    print("=" * 70)
    print("FETCHING ARGO DATA")
    print("=" * 70)

    print(
        f"Region      : "
        f"{ARGO_LON_MIN}–{ARGO_LON_MAX}°E, "
        f"{ARGO_LAT_MIN}–{ARGO_LAT_MAX}°N"
    )

    print(
        f"Period      : "
        f"{ARGO_START} -> {ARGO_END}"
    )

    print(
        "Depth range : 0–1000 m"
    )

    fetcher = (
        argopy
        .DataFetcher(
            src="erddap"
        )
        .region(
            [
                ARGO_LON_MIN,
                ARGO_LON_MAX,
                ARGO_LAT_MIN,
                ARGO_LAT_MAX,
                0,
                1000,
                ARGO_START,
                ARGO_END,
            ]
        )
    )

    ds = fetcher.to_dataset()

    print()
    print(
        "ARGO variables:",
        list(ds.variables.keys()),
    )

    temp = np.asarray(
        ds.variables["TEMP"][:],
        dtype=np.float64,
    )

    pres = np.asarray(
        ds.variables["PRES"][:],
        dtype=np.float64,
    )

    temp_qc = np.asarray(
        ds.variables["TEMP_QC"][:]
    )

    lat = np.asarray(
        ds.variables["LATITUDE"][:],
        dtype=np.float64,
    )

    lon = np.asarray(
        ds.variables["LONGITUDE"][:],
        dtype=np.float64,
    )

    wmo = np.asarray(
        ds.variables["PLATFORM_NUMBER"][:]
    )

    cycle = np.asarray(
        ds.variables["CYCLE_NUMBER"][:]
    )

    time_variable = ds.variables["TIME"]

    time_values = np.asarray(
        time_variable[:]
    )

    print()
    print(
        "Total ARGO observations:",
        len(temp),
    )

    return {
        "temp": temp,
        "pres": pres,
        "temp_qc": temp_qc,
        "lat": lat,
        "lon": lon,
        "wmo": wmo,
        "cycle": cycle,
        "time": time_values,
        "time_variable": time_variable,
    }


# ============================================================
# BUILD PROFILES
# ============================================================

def build_profiles(argo):
    print()
    print("=" * 70)
    print("BUILDING ARGO PROFILES")
    print("=" * 70)

    temp = argo["temp"]
    pres = argo["pres"]
    temp_qc = argo["temp_qc"]

    wmo = argo["wmo"]
    cycle = argo["cycle"]

    lat = argo["lat"]
    lon = argo["lon"]

    time_values = argo["time"]
    time_variable = argo["time_variable"]

    groups = {}

    for i in range(
        len(temp)
    ):

        try:
            qc_value = int(
                temp_qc[i]
            )
        except Exception:
            continue

        if qc_value not in (
            1,
            2,
        ):
            continue

        if not np.isfinite(
            temp[i]
        ):
            continue

        if not np.isfinite(
            pres[i]
        ):
            continue

        key = (
            str(wmo[i]),
            int(cycle[i]),
        )

        if key not in groups:
            groups[key] = {
                "pressure": [],
                "temperature": [],
                "latitude": float(
                    lat[i]
                ),
                "longitude": float(
                    lon[i]
                ),
                "time": convert_time_value(
                    time_values[i],
                    time_variable,
                ),
                "wmo": str(
                    wmo[i]
                ),
                "cycle": int(
                    cycle[i]
                ),
            }

        groups[key][
            "pressure"
        ].append(
            float(
                pres[i]
            )
        )

        groups[key][
            "temperature"
        ].append(
            float(
                temp[i]
            )
        )

    profiles = list(
        groups.values()
    )

    print(
        "Profiles after QC filtering:",
        len(profiles),
    )

    profiles.sort(
        key=lambda item: item["time"]
    )

    if MAX_PROFILES is not None:
        profiles = profiles[
            :MAX_PROFILES
        ]

    print(
        "Profiles selected for pilot:",
        len(profiles),
    )

    return profiles


# ============================================================
# INTERPOLATE ARGO PROFILE
# ============================================================

def interpolate_profile(
    profile,
):
    pressure = np.asarray(
        profile["pressure"],
        dtype=np.float64,
    )

    temperature = np.asarray(
        profile["temperature"],
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            pressure
        )
        &
        np.isfinite(
            temperature
        )
    )

    pressure = pressure[
        valid
    ]

    temperature = temperature[
        valid
    ]

    if len(pressure) < 5:
        return None

    order = np.argsort(
        pressure
    )

    pressure = pressure[
        order
    ]

    temperature = temperature[
        order
    ]

    unique_pressure, unique_indices = (
        np.unique(
            pressure,
            return_index=True,
        )
    )

    unique_temperature = (
        temperature[
            unique_indices
        ]
    )

    pressure = unique_pressure
    temperature = unique_temperature

    observed = np.full(
        len(DEPTHS),
        np.nan,
        dtype=np.float64,
    )

    valid_depths = (
        (DEPTHS >= pressure.min())
        &
        (DEPTHS <= pressure.max())
    )

    if not np.any(
        valid_depths
    ):
        return None

    observed[
        valid_depths
    ] = np.interp(
        DEPTHS[
            valid_depths
        ],
        pressure,
        temperature,
    )

    return observed


# ============================================================
# EXTRACT 9x9 PATCH
# ============================================================

def extract_patch(
    surface_raw,
    block_start_date,
    profile_date,
    latitude,
    longitude,
):
    """
    surface_raw shape:
        [days, 7, lat, lon]
    """

    if not isinstance(
        surface_raw,
        np.ndarray,
    ):
        surface_raw = np.asarray(
            surface_raw,
            dtype=np.float32,
        )

    date_delta = (
        profile_date
        - block_start_date
    ).days

    if date_delta < 0:
        raise ValueError(
            "Profile date occurs before "
            "loaded surface block."
        )

    if date_delta >= surface_raw.shape[0]:
        raise ValueError(
            "Profile date occurs after "
            "loaded surface block."
        )

    day_index = date_delta

    lat_index = int(
        np.argmin(
            np.abs(
                np.asarray(
                    LATS
                )
                - latitude
            )
        )
    )

    lon_index = int(
        np.argmin(
            np.abs(
                np.asarray(
                    LONS
                )
                - longitude
            )
        )
    )

    lat_index = int(
        np.clip(
            lat_index,
            4,
            len(LATS) - 5,
        )
    )

    lon_index = int(
        np.clip(
            lon_index,
            4,
            len(LONS) - 5,
        )
    )

    patch = surface_raw[
        day_index,
        :,
        lat_index - 4:
        lat_index + 5,
        lon_index - 4:
        lon_index + 5,
    ]

    if patch.shape != (
        7,
        9,
        9,
    ):
        raise ValueError(
            f"Unexpected patch shape: "
            f"{patch.shape}"
        )

    return patch


# ============================================================
# PREDICT
# ============================================================

def predict_profile(
    model,
    target_mean,
    target_std,
    patch_raw,
):
    patch_batch = (
        patch_raw[
            np.newaxis,
            ...
        ]
    )

    patch_norm = normalize_surface(
        patch_batch
    )

    x = torch.from_numpy(
        patch_norm.astype(
            np.float32
        )
    ).to(
        DEVICE
    )

    with torch.no_grad():
        prediction_normalized = (
            model(x)
            .cpu()
            .numpy()[0]
        )

    prediction_celsius = (
        prediction_normalized
        * target_std
        + target_mean
    )

    return prediction_celsius.astype(
        np.float64
    )


# ============================================================
# LOAD 30-DAY SURFACE BLOCK
# ============================================================

def load_surface_for_date(
    date_text,
    cache,
):
    date_obj = datetime.strptime(
        date_text,
        "%Y-%m-%d",
    ).date()

    block_start = date_obj

    block_end = (
        block_start
        + timedelta(
            days=29
        )
    )

    cache_key = (
        block_start.isoformat(),
        block_end.isoformat(),
    )

    if cache_key in cache:
        return (
            cache[cache_key],
            block_start,
        )

    print()
    print("-" * 70)
    print(
        "LOADING ONLINE SURFACE BLOCK:",
        block_start,
        "->",
        block_end,
    )
    print("-" * 70)

    # IMPORTANT:
    # load_surface_block returns:
    #
    #   (surface_raw, surface_normalized)
    #
    # We use surface_raw because the existing OceanEmbed
    # normalization is applied before model inference.
    surface_raw, surface_normalized = (
        load_surface_block(
            block_start.isoformat(),
            block_end.isoformat(),
        )
    )

    # We don't need surface_normalized here.
    # Keeping it in the unpacking makes the return structure explicit.
    del surface_normalized

    cache[cache_key] = surface_raw

    return (
        surface_raw,
        block_start,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    predictions,
    observations,
):
    predictions = np.asarray(
        predictions,
        dtype=np.float64,
    )

    observations = np.asarray(
        observations,
        dtype=np.float64,
    )

    valid = (
        np.isfinite(
            predictions
        )
        &
        np.isfinite(
            observations
        )
    )

    if not np.any(
        valid
    ):
        return None

    error = (
        predictions[valid]
        -
        observations[valid]
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    bias = np.mean(
        error
    )

    return {
        "count": int(
            np.sum(valid)
        ),
        "mae": float(
            mae
        ),
        "rmse": float(
            rmse
        ),
        "bias": float(
            bias
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "OCEANEMBED - INDEPENDENT ARGO PILOT VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "ARGO is used ONLY for independent evaluation."
    )

    print(
        "The model is NOT retrained with ARGO data."
    )

    print(
        "Training remains based on 2023 GLORYS."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    (
        model,
        target_mean,
        target_std,
    ) = load_oceanembed_model()

    # --------------------------------------------------------
    # FETCH ARGO
    # --------------------------------------------------------

    argo = fetch_argo()

    profiles = build_profiles(
        argo
    )

    if len(profiles) == 0:
        raise RuntimeError(
            "No valid ARGO profiles remained "
            "after QC filtering."
        )

    # --------------------------------------------------------
    # PROCESS PROFILES
    # --------------------------------------------------------

    surface_cache = {}

    result_rows = []

    all_predictions = []
    all_observations = []

    print()
    print("=" * 70)
    print(
        "RUNNING ARGO -> OCEANEMBED VALIDATION"
    )
    print("=" * 70)

    for profile_number, profile in enumerate(
        profiles,
        start=1,
    ):

        profile_date = (
            profile["time"].date()
        )

        date_text = (
            profile_date.isoformat()
        )

        print()
        print(
            f"[{profile_number}/{len(profiles)}] "
            f"WMO {profile['wmo']} "
            f"Cycle {profile['cycle']}"
        )

        print(
            "Location:",
            f"{profile['latitude']:.3f}°N, "
            f"{profile['longitude']:.3f}°E",
        )

        print(
            "Date    :",
            date_text,
        )

        # ----------------------------------------------------
        # ARGO OBSERVATION
        # ----------------------------------------------------

        observed_profile = (
            interpolate_profile(
                profile
            )
        )

        if observed_profile is None:
            print(
                "Skipped: insufficient "
                "ARGO vertical coverage."
            )
            continue

        observed_count = int(
            np.sum(
                np.isfinite(
                    observed_profile
                )
            )
        )

        print(
            "Observed OceanEmbed depths:",
            observed_count,
            "/",
            len(DEPTHS),
        )

        # ----------------------------------------------------
        # LOAD SURFACE DATA
        # ----------------------------------------------------

        try:
            (
                surface_raw,
                block_start,
            ) = load_surface_for_date(
                date_text,
                surface_cache,
            )

            patch = extract_patch(
                surface_raw,
                block_start,
                profile_date,
                profile[
                    "latitude"
                ],
                profile[
                    "longitude"
                ],
            )

        except Exception as exc:

            print(
                "Skipped because surface-data retrieval "
                "or patch extraction failed:"
            )

            print(
                repr(exc)
            )

            continue

        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = predict_profile(
            model,
            target_mean,
            target_std,
            patch,
        )

        valid_depths = (
            np.isfinite(
                observed_profile
            )
            &
            np.isfinite(
                prediction
            )
        )

        # ----------------------------------------------------
        # STORE DEPTH RESULTS
        # ----------------------------------------------------

        for depth, pred, obs in zip(
            DEPTHS,
            prediction,
            observed_profile,
        ):

            if not (
                np.isfinite(pred)
                and np.isfinite(obs)
            ):
                continue

            error = (
                float(pred)
                -
                float(obs)
            )

            result_rows.append(
                {
                    "wmo": profile[
                        "wmo"
                    ],
                    "cycle": profile[
                        "cycle"
                    ],
                    "date": date_text,
                    "latitude": profile[
                        "latitude"
                    ],
                    "longitude": profile[
                        "longitude"
                    ],
                    "depth_m": float(
                        depth
                    ),
                    "observed_temp_c": float(
                        obs
                    ),
                    "predicted_temp_c": float(
                        pred
                    ),
                    "error_c": error,
                    "absolute_error_c": abs(
                        error
                    ),
                }
            )

        all_predictions.extend(
            prediction[
                valid_depths
            ]
        )

        all_observations.extend(
            observed_profile[
                valid_depths
            ]
        )

        profile_metrics = calculate_metrics(
            prediction[
                valid_depths
            ],
            observed_profile[
                valid_depths
            ],
        )

        if profile_metrics is not None:

            print(
                f"Profile MAE  : "
                f"{profile_metrics['mae']:.4f} °C"
            )

            print(
                f"Profile RMSE : "
                f"{profile_metrics['rmse']:.4f} °C"
            )

            print(
                f"Profile Bias : "
                f"{profile_metrics['bias']:.4f} °C"
            )

    # --------------------------------------------------------
    # OVERALL METRICS
    # --------------------------------------------------------

    if len(
        all_predictions
    ) == 0:

        raise RuntimeError(
            "No ARGO profiles produced "
            "a valid prediction."
        )

    all_predictions = np.asarray(
        all_predictions,
        dtype=np.float64,
    )

    all_observations = np.asarray(
        all_observations,
        dtype=np.float64,
    )

    overall = calculate_metrics(
        all_predictions,
        all_observations,
    )

    # --------------------------------------------------------
    # DEPTHWISE METRICS
    # --------------------------------------------------------

    result_df = pd.DataFrame(
        result_rows
    )

    depth_metrics = []

    if not result_df.empty:

        for depth in DEPTHS:

            subset = result_df[
                result_df[
                    "depth_m"
                ]
                ==
                float(depth)
            ]

            if subset.empty:

                depth_metrics.append(
                    {
                        "depth_m": float(
                            depth
                        ),
                        "count": 0,
                        "mae_c": np.nan,
                        "rmse_c": np.nan,
                        "bias_c": np.nan,
                    }
                )

                continue

            observed = subset[
                "observed_temp_c"
            ].to_numpy(
                dtype=np.float64
            )

            predicted = subset[
                "predicted_temp_c"
            ].to_numpy(
                dtype=np.float64
            )

            metrics = calculate_metrics(
                predicted,
                observed,
            )

            depth_metrics.append(
                {
                    "depth_m": float(
                        depth
                    ),
                    "count": metrics[
                        "count"
                    ],
                    "mae_c": metrics[
                        "mae"
                    ],
                    "rmse_c": metrics[
                        "rmse"
                    ],
                    "bias_c": metrics[
                        "bias"
                    ],
                }
            )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    result_df.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    unique_profiles = (
        result_df[
            [
                "wmo",
                "cycle",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        if not result_df.empty
        else 0
    )

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "OceanEmbed Independent ARGO Pilot Validation\n"
        )

        f.write(
            "=" * 70 + "\n"
        )

        f.write(
            "Training data: 2023 GLORYS\n"
        )

        f.write(
            "Evaluation data: 2024 ARGO observations\n"
        )

        f.write(
            "Model: OceanEmbed Embedding V2\n"
        )

        f.write(
            f"Region: "
            f"{ARGO_LAT_MIN}-{ARGO_LAT_MAX} N, "
            f"{ARGO_LON_MIN}-{ARGO_LON_MAX} E\n"
        )

        f.write(
            f"Period: "
            f"{ARGO_START} -> {ARGO_END}\n"
        )

        f.write(
            f"Profiles evaluated: "
            f"{unique_profiles}\n"
        )

        f.write(
            f"Depth observations: "
            f"{overall['count']}\n"
        )

        f.write(
            f"Overall MAE: "
            f"{overall['mae']:.6f} C\n"
        )

        f.write(
            f"Overall RMSE: "
            f"{overall['rmse']:.6f} C\n"
        )

        f.write(
            f"Overall Bias: "
            f"{overall['bias']:.6f} C\n"
        )

        f.write(
            "\nDepthwise metrics\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for item in depth_metrics:

            if item["count"] == 0:

                f.write(
                    f"{item['depth_m']:6.0f} m | "
                    f"no observations\n"
                )

            else:

                f.write(
                    f"{item['depth_m']:6.0f} m | "
                    f"N={item['count']:4d} | "
                    f"MAE={item['mae_c']:.4f} | "
                    f"RMSE={item['rmse_c']:.4f} | "
                    f"Bias={item['bias_c']:.4f}\n"
                )

    # --------------------------------------------------------
    # FINAL CONSOLE SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "ARGO PILOT VALIDATION COMPLETE"
    )
    print("=" * 70)

    print(
        "Profiles evaluated:",
        unique_profiles,
    )

    print(
        "Valid depth comparisons:",
        overall["count"],
    )

    print()

    print(
        f"Overall MAE  : "
        f"{overall['mae']:.4f} °C"
    )

    print(
        f"Overall RMSE : "
        f"{overall['rmse']:.4f} °C"
    )

    print(
        f"Overall Bias : "
        f"{overall['bias']:.4f} °C"
    )

    print()
    print(
        "Detailed CSV:"
    )
    print(
        CSV_OUTPUT
    )

    print()
    print(
        "Summary:"
    )
    print(
        SUMMARY_OUTPUT
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()