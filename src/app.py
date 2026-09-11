import os
import sys
from functools import lru_cache

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# OCEANEMBED V2
# ============================================================

from model_embed import OceanEmbedEmbeddingModel

from online_year_data import (
    load_surface_block,
    load_surface_patch,
    normalize_surface,
)

from ocean_mask import check_location, get_ocean_mask
from config import LATS, LONS


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="templates",
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "oceanembed_embedding_v2_2023.pt",
)


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


DEVICE = torch.device("cpu")


# ============================================================
# CROSS-SECTION CONFIGURATION
# ============================================================

CROSS_SECTION_POINTS = 49
CROSS_SECTION_LON_MIN = 46.0
CROSS_SECTION_LON_MAX = 104.0


# ============================================================
# SURFACE FEATURES
# ============================================================

FEATURE_NAMES = [
    "SST",
    "SSS",
    "SSH / SLA",
    "U Current",
    "V Current",
    "U Wind",
    "V Wind",
]


FEATURE_UNITS = [
    "°C",
    "PSU",
    "m",
    "m/s",
    "m/s",
    "m/s",
    "m/s",
]


# ============================================================
# MODEL STATE
# ============================================================

model = None
target_mean = None
target_std = None


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def _to_numpy(value):
    """
    Convert checkpoint values to NumPy arrays.
    """

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()

    return np.asarray(value, dtype=np.float32)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    global model
    global target_mean
    global target_std

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            "V2 trained model not found:\n"
            + CHECKPOINT_PATH
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = OceanEmbedEmbeddingModel().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    if "target_mean" not in checkpoint:
        raise KeyError(
            "V2 checkpoint does not contain target_mean."
        )

    if "target_std" not in checkpoint:
        raise KeyError(
            "V2 checkpoint does not contain target_std."
        )

    target_mean = _to_numpy(
        checkpoint["target_mean"]
    ).reshape(-1)

    target_std = _to_numpy(
        checkpoint["target_std"]
    ).reshape(-1)

    if len(target_mean) != 15:
        raise ValueError(
            f"Expected 15 target means, got {len(target_mean)}."
        )

    if len(target_std) != 15:
        raise ValueError(
            f"Expected 15 target standard deviations, got {len(target_std)}."
        )

    print("OceanEmbed V2 model loaded.")
    print("Parameters: 49,007")
    print("Embedding dimension: 64")


# ============================================================
# BASIC HELPERS
# ============================================================

def _validate_date(selected_date):
    return (
        "2023-01-01"
        <= selected_date
        <= "2024-12-15"
    )


def _denormalize_temperature(
    prediction_normalized
):
    """
    Convert normalized V2 predictions
    back to physical temperature in °C.
    """

    prediction_normalized = np.asarray(
        prediction_normalized,
        dtype=np.float64,
    )

    return (
        prediction_normalized
        * target_std
        + target_mean
    )


# ============================================================
# SINGLE PATCH PREDICTION
# ============================================================

def _prediction_from_patch(patch_raw):

    if model is None:
        load_model()

    patch_raw_batched = (
        patch_raw[np.newaxis, ...]
    )

    patch_norm = normalize_surface(
        patch_raw_batched
    )

    x = torch.from_numpy(
        patch_norm.astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():

        prediction_normalized = (
            model(x)
            .cpu()
            .numpy()[0]
        )

    prediction = _denormalize_temperature(
        prediction_normalized
    )

    return prediction.astype(np.float64)


# ============================================================
# BATCH PREDICTION
# ============================================================

def _prediction_batch_from_patches(
    patches_raw
):

    if model is None:
        load_model()

    patches_norm = normalize_surface(
        patches_raw
    )

    x = torch.from_numpy(
        patches_norm.astype(np.float32)
    ).to(DEVICE)

    with torch.no_grad():

        prediction_normalized = (
            model(x)
            .cpu()
            .numpy()
        )

    predictions = _denormalize_temperature(
        prediction_normalized
    )

    return predictions.astype(np.float64)


# ============================================================
# PROFILE JSON
# ============================================================

def _profile_json(prediction):

    profile = []

    for depth, temperature in zip(
        DEPTHS,
        prediction,
    ):

        profile.append(
            {
                "depth_m": float(depth),
                "temperature_c": round(
                    float(temperature),
                    3,
                ),
            }
        )

    return profile


# ============================================================
# SAFE FLOAT
# ============================================================

def _safe_float(value):

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    return value


# ============================================================
# SURFACE INPUTS
# ============================================================

def _surface_inputs_from_patch(
    patch_raw
):

    center_values = patch_raw[:, 4, 4]

    return [
        {
            "name": name,
            "unit": unit,
            "value": _safe_float(value),
        }
        for name, unit, value in zip(
            FEATURE_NAMES,
            FEATURE_UNITS,
            center_values,
        )
    ]


# ============================================================
# EXTRACT 9x9 PATCH
# ============================================================

def _extract_surface_patch(
    surface_day,
    lat_index,
    lon_index,
):

    _, n_lat, n_lon = surface_day.shape

    lat_index = int(
        np.clip(
            lat_index,
            4,
            n_lat - 5,
        )
    )

    lon_index = int(
        np.clip(
            lon_index,
            4,
            n_lon - 5,
        )
    )

    return surface_day[
        :,
        lat_index - 4 : lat_index + 5,
        lon_index - 4 : lon_index + 5,
    ]


# ============================================================
# CROSS-SECTION LONGITUDES
# ============================================================

def _make_cross_section_longitudes():

    candidates = np.linspace(
        CROSS_SECTION_LON_MIN,
        CROSS_SECTION_LON_MAX,
        CROSS_SECTION_POINTS,
    )

    indices = []

    for lon in candidates:

        index = int(
            np.argmin(
                np.abs(
                    np.asarray(LONS)
                    - lon
                )
            )
        )

        if index < 4:
            continue

        if index > len(LONS) - 5:
            continue

        if index not in indices:
            indices.append(index)

    return indices


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"],
)
def health():

    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,

            "model": {
                "name": "OceanEmbed Embedding V2",
                "parameters": 49007,
                "embedding_dimension": 64,
                "training_year": 2023,
                "validation_year": 2024,
            },

            "validation": {
                "mae_c": 0.7502,
                "rmse_c": 1.1132,
            },

            "domain": {
                "lat_min": 5,
                "lat_max": 30,
                "lon_min": 45,
                "lon_max": 105,
            },

            "depths": [
                float(x)
                for x in DEPTHS
            ],

            "inputs": FEATURE_NAMES,
        }
    )


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"],
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify(
                {
                    "success": False,
                    "error": "Request must contain JSON.",
                }
            ), 400

        latitude = float(
            data.get("latitude")
        )

        longitude = float(
            data.get("longitude")
        )

        selected_date = str(
            data.get("date")
        )

        # ----------------------------------------------------
        # DOMAIN
        # ----------------------------------------------------

        if not (
            5 <= latitude <= 30
            and
            45 <= longitude <= 105
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "OUT_OF_DOMAIN",
                    "message": (
                        "Selected location is outside "
                        "the North Indian Ocean model domain."
                    ),
                }
            ), 400

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if not _validate_date(
            selected_date
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "INVALID_DATE",
                    "message": (
                        "Please select a date between "
                        "2023-01-01 and 2024-12-15."
                    ),
                }
            ), 400

        # ----------------------------------------------------
        # LAND / OCEAN
        # ----------------------------------------------------

        location = check_location(
            latitude,
            longitude,
        )

        if not location["is_ocean"]:

            return jsonify(
                {
                    "success": False,
                    "type": location["status"],
                    "message": location["message"],
                }
            ), 400

        # ----------------------------------------------------
        # ONLINE SURFACE DATA
        # ----------------------------------------------------

        print()
        print("=" * 70)
        print("NEW OCEANEMBED V2 REQUEST")
        print(
            f"Location: "
            f"{latitude:.4f}, "
            f"{longitude:.4f}"
        )
        print(
            f"Date: {selected_date}"
        )
        print(
            "Fetching online surface data..."
        )

        patch_raw = load_surface_patch(
            selected_date,
            latitude,
            longitude,
        )

        # ----------------------------------------------------
        # V2 MODEL
        # ----------------------------------------------------

        prediction = _prediction_from_patch(
            patch_raw
        )

        profile = _profile_json(
            prediction
        )

        surface_inputs = (
            _surface_inputs_from_patch(
                patch_raw
            )
        )

        print(
            "OceanEmbed V2 prediction complete."
        )
        print("=" * 70)

        return jsonify(
            {
                "success": True,

                "location": {
                    "latitude": latitude,
                    "longitude": longitude,
                },

                "date": selected_date,

                "nearest_grid": {
                    "latitude": location.get(
                        "nearest_grid_latitude"
                    ),
                    "longitude": location.get(
                        "nearest_grid_longitude"
                    ),
                },

                "profile": profile,

                "surface_inputs": surface_inputs,

                "model_input": {
                    "channels": 7,
                    "patch_shape": [
                        7,
                        9,
                        9,
                    ],
                },

                "model_info": {
                    "name": "OceanEmbed Embedding V2",
                    "parameters": 49007,
                    "embedding_dimension": 64,
                    "training_year": 2023,
                    "validation_year": 2024,
                    "validation_mae_c": 0.7502,
                    "validation_rmse_c": 1.1132,
                },
            }
        )

    except (TypeError, ValueError) as e:

        return jsonify(
            {
                "success": False,
                "type": "INVALID_REQUEST",
                "error": str(e),
            }
        ), 400

    except Exception as e:

        print()
        print("PREDICTION ERROR:")
        print(repr(e))

        return jsonify(
            {
                "success": False,
                "type": "SERVER_ERROR",
                "error": str(e),
            }
        ), 500


# ============================================================
# CROSS-SECTION CACHE
# ============================================================

@lru_cache(maxsize=6)
def _cross_section_cached(
    date_text,
    latitude_rounded,
):

    latitude = float(
        latitude_rounded
    )

    # --------------------------------------------------------
    # FIND LATITUDE GRID
    # --------------------------------------------------------

    lat_index = int(
        np.argmin(
            np.abs(
                np.asarray(LATS)
                - latitude
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

    section_latitude = float(
        LATS[lat_index]
    )

    lon_indices = (
        _make_cross_section_longitudes()
    )

    # --------------------------------------------------------
    # FETCH FULL DAY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BUILDING V2 THERMAL CROSS-SECTION")
    print(
        f"Date: {date_text}"
    )
    print(
        f"Requested latitude: "
        f"{latitude:.3f}"
    )
    print(
        f"Grid latitude: "
        f"{section_latitude:.3f}"
    )
    print(
        "Fetching one full-day surface block..."
    )

    _, surface_raw, dates = (
        load_surface_block(
            date_text,
            date_text,
        )
    )

    if len(dates) == 0:
        raise ValueError(
            "No surface data returned "
            "for the selected date."
        )

    surface_day = np.asarray(
        surface_raw[0],
        dtype=np.float32,
    )

    if surface_day.shape[0] != 7:

        raise ValueError(
            "Unexpected number of surface "
            f"channels: {surface_day.shape}"
        )

    ocean_mask = get_ocean_mask()

    patches = []
    valid_lon_indices = []

    # --------------------------------------------------------
    # BUILD PATCHES
    # --------------------------------------------------------

    for lon_index in lon_indices:

        if not bool(
            ocean_mask[
                lat_index,
                lon_index
            ]
        ):
            continue

        patch = _extract_surface_patch(
            surface_day,
            lat_index,
            lon_index,
        )

        if patch.shape != (
            7,
            9,
            9,
        ):
            continue

        patches.append(patch)

        valid_lon_indices.append(
            lon_index
        )

    if not patches:

        raise ValueError(
            "No valid ocean points were found "
            "for this cross-section latitude."
        )

    # --------------------------------------------------------
    # V2 BATCH INFERENCE
    # --------------------------------------------------------

    patch_array = np.stack(
        patches,
        axis=0,
    )

    predictions = (
        _prediction_batch_from_patches(
            patch_array
        )
    )

    # --------------------------------------------------------
    # DEPTH × LONGITUDE MATRIX
    # --------------------------------------------------------

    longitudes = [
        float(LONS[index])
        for index in valid_lon_indices
    ]

    temperature_matrix = (
        predictions
        .T
        .tolist()
    )

    print(
        "V2 cross-section complete: "
        f"{len(longitudes)} ocean longitudes."
    )

    print("=" * 70)

    return {
        "success": True,
        "date": date_text,
        "requested_latitude": float(
            latitude
        ),
        "section_latitude": section_latitude,
        "longitudes": longitudes,
        "depths": [
            float(x)
            for x in DEPTHS
        ],
        "temperature": temperature_matrix,
    }


# ============================================================
# CROSS-SECTION API
# ============================================================

@app.route(
    "/api/cross-section",
    methods=["POST"],
)
def cross_section():

    try:

        data = request.get_json(
            silent=True
        )

        if data is None:

            return jsonify(
                {
                    "success": False,
                    "error": "Request must contain JSON.",
                }
            ), 400

        latitude = float(
            data.get("latitude")
        )

        longitude = float(
            data.get("longitude")
        )

        selected_date = str(
            data.get("date")
        )

        # ----------------------------------------------------
        # DOMAIN
        # ----------------------------------------------------

        if not (
            5 <= latitude <= 30
            and
            45 <= longitude <= 105
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "OUT_OF_DOMAIN",
                    "message": (
                        "Selected location is outside "
                        "the North Indian Ocean model domain."
                    ),
                }
            ), 400

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if not _validate_date(
            selected_date
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "INVALID_DATE",
                    "message": (
                        "Please select a date between "
                        "2023-01-01 and 2024-12-15."
                    ),
                }
            ), 400

        # ----------------------------------------------------
        # OCEAN CHECK
        # ----------------------------------------------------

        location = check_location(
            latitude,
            longitude,
        )

        if not location["is_ocean"]:

            return jsonify(
                {
                    "success": False,
                    "type": location["status"],
                    "message": location["message"],
                }
            ), 400

        # ----------------------------------------------------
        # CACHE
        # ----------------------------------------------------

        cache_latitude = round(
            latitude,
            4,
        )

        result = _cross_section_cached(
            selected_date,
            cache_latitude,
        )

        result["selected_longitude"] = (
            longitude
        )

        result["selected_grid_longitude"] = (
            location.get(
                "nearest_grid_longitude"
            )
        )

        return jsonify(result)

    except (TypeError, ValueError) as e:

        return jsonify(
            {
                "success": False,
                "type": "INVALID_REQUEST",
                "error": str(e),
            }
        ), 400

    except Exception as e:

        print()
        print("CROSS-SECTION ERROR:")
        print(repr(e))

        return jsonify(
            {
                "success": False,
                "type": "SERVER_ERROR",
                "error": str(e),
            }
        ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    load_model()

    print()
    print("=" * 70)
    print("OCEANEMBED V2 WEB APPLICATION")
    print("=" * 70)
    print("Model: OceanEmbed Embedding V2")
    print("Parameters: 49,007")
    print("Embedding dimension: 64")
    print("Domain: 5-30 N, 45-105 E")
    print("Training: 2023")
    print("Validation: 2024")
    print("MAE: 0.7502 C")
    print("RMSE: 1.1132 C")
    print("Surface data: ONLINE")
    print("Thermal cross-section: ENABLED")
    print("Server: http://127.0.0.1:5000")
    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )