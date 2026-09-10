import os
from functools import lru_cache

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# IMPORT OCEANEMBED COMPONENTS
# ============================================================
#
# IMPORTANT:
# These are imported through the src package because
# ocean_mask.py and the other src files use relative imports.
#
# DO NOT change these to:
# from ocean_mask import ...
#
# They MUST remain:
# from src.ocean_mask import ...
# ============================================================

from src.model_fast import OceanEmbedFast

from src.online_year_data import (
    load_surface_patch,
    load_surface_block,
    normalize_surface,
)

from src.ocean_mask import (
    check_location,
    get_ocean_mask,
)

from src.config import (
    LATS,
    LONS,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__,
    template_folder="templates"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "oceanembed_fast_2023.pt"
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
    dtype=np.float32
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# SURFACE FEATURE CONFIGURATION
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
# CROSS-SECTION CONFIGURATION
# ============================================================

CROSS_SECTION_POINTS = 49

CROSS_SECTION_LON_MIN = 46.0

CROSS_SECTION_LON_MAX = 104.0


# ============================================================
# GLOBAL MODEL
# ============================================================

model = None


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    global model

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "Trained model not found:\n"
            + CHECKPOINT_PATH
        )


    model = OceanEmbedFast().to(
        DEVICE
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    model.load_state_dict(
        checkpoint["model_state"]
    )


    model.eval()


    print(
        "OceanEmbed model loaded successfully."
    )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_date(
    date_text
):

    return (
        "2023-01-01"
        <= date_text
        <= "2024-12-15"
    )


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    try:

        value = float(
            value
        )

    except (
        TypeError,
        ValueError
    ):

        return None


    if not np.isfinite(
        value
    ):

        return None


    return value


# ============================================================
# PREDICT FROM ONE 7x9x9 PATCH
# ============================================================

def predict_from_patch(
    patch_raw
):

    patch_batched = patch_raw[
        np.newaxis,
        ...
    ]


    patch_normalized = normalize_surface(
        patch_batched
    )


    x = torch.from_numpy(
        patch_normalized.astype(
            np.float32
        )
    ).to(
        DEVICE
    )


    if model is None:

        load_model()


    with torch.no_grad():

        prediction = model(
            x
        ).cpu().numpy()[0]


    return prediction.astype(
        np.float64
    )


# ============================================================
# CONVERT PROFILE TO JSON
# ============================================================

def profile_to_json(
    prediction
):

    profile = []


    for depth, temperature in zip(
        DEPTHS,
        prediction
    ):

        profile.append(
            {
                "depth_m":
                    float(
                        depth
                    ),

                "temperature_c":
                    round(
                        float(
                            temperature
                        ),
                        3
                    ),
            }
        )


    return profile


# ============================================================
# EXTRACT ACTUAL SURFACE INPUT VALUES
# ============================================================

def extract_surface_inputs(
    patch_raw
):

    # Center grid cell of the 9x9 model patch
    # is position [4,4].

    center_values = patch_raw[
        :,
        4,
        4
    ]


    result = []


    for name, unit, value in zip(
        FEATURE_NAMES,
        FEATURE_UNITS,
        center_values
    ):

        result.append(
            {
                "name":
                    name,

                "unit":
                    unit,

                "value":
                    safe_float(
                        value
                    ),
            }
        )


    return result


# ============================================================
# EXTRACT 9x9 PATCH FROM FULL SURFACE DAY
# ============================================================

def extract_surface_patch(
    surface_day,
    lat_index,
    lon_index
):

    # surface_day:
    # [7, latitude, longitude]

    _, n_lat, n_lon = (
        surface_day.shape
    )


    lat_index = int(
        np.clip(
            lat_index,
            4,
            n_lat - 5
        )
    )


    lon_index = int(
        np.clip(
            lon_index,
            4,
            n_lon - 5
        )
    )


    patch = surface_day[
        :,
        lat_index - 4:
        lat_index + 5,
        lon_index - 4:
        lon_index + 5,
    ]


    return patch


# ============================================================
# CROSS-SECTION LONGITUDES
# ============================================================

def get_cross_section_longitudes():

    candidates = np.linspace(
        CROSS_SECTION_LON_MIN,
        CROSS_SECTION_LON_MAX,
        CROSS_SECTION_POINTS
    )


    indices = []


    for longitude in candidates:

        index = int(
            np.argmin(
                np.abs(
                    np.asarray(LONS)
                    - longitude
                )
            )
        )


        if index < 4:

            continue


        if index > len(LONS) - 5:

            continue


        if index not in indices:

            indices.append(
                index
            )


    return indices


# ============================================================
# HOME PAGE
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
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status":
                "ok",

            "model_loaded":
                model is not None,

            "domain":
                {
                    "lat_min":
                        5,

                    "lat_max":
                        30,

                    "lon_min":
                        45,

                    "lon_max":
                        105,
                },

            "depths":
                [
                    float(x)
                    for x in DEPTHS
                ],

            "inputs":
                FEATURE_NAMES,

            "cross_section":
                True,
        }
    )


# ============================================================
# SINGLE POINT PREDICTION
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify(
                {
                    "success":
                        False,

                    "error":
                        "Request must contain JSON.",
                }
            ), 400


        latitude = float(
            data.get(
                "latitude"
            )
        )


        longitude = float(
            data.get(
                "longitude"
            )
        )


        selected_date = str(
            data.get(
                "date"
            )
        )


        # ----------------------------------------------------
        # DOMAIN CHECK
        # ----------------------------------------------------

        if not (
            5 <= latitude <= 30
            and
            45 <= longitude <= 105
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "OUT_OF_DOMAIN",

                    "message":
                        "Selected location is outside "
                        "the North Indian Ocean model domain.",
                }
            ), 400


        # ----------------------------------------------------
        # DATE CHECK
        # ----------------------------------------------------

        if not validate_date(
            selected_date
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "INVALID_DATE",

                    "message":
                        "Please select a date between "
                        "2023-01-01 and 2024-12-15.",
                }
            ), 400


        # ----------------------------------------------------
        # LAND / OCEAN CHECK
        # ----------------------------------------------------

        location = check_location(
            latitude,
            longitude
        )


        if not location[
            "is_ocean"
        ]:

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        location[
                            "status"
                        ],

                    "message":
                        location[
                            "message"
                        ],
                }
            ), 400


        # ----------------------------------------------------
        # ONLINE SURFACE DATA
        # ----------------------------------------------------

        print()
        print(
            "=" * 70
        )

        print(
            "NEW OCEANEMBED REQUEST"
        )

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
            longitude
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = predict_from_patch(
            patch_raw
        )


        # ----------------------------------------------------
        # PROFILE
        # ----------------------------------------------------

        profile = profile_to_json(
            prediction
        )


        # ----------------------------------------------------
        # SURFACE INPUT VALUES
        # ----------------------------------------------------

        surface_inputs = extract_surface_inputs(
            patch_raw
        )


        print(
            "Prediction complete."
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify(
            {
                "success":
                    True,

                "location":
                    {
                        "latitude":
                            latitude,

                        "longitude":
                            longitude,
                    },

                "date":
                    selected_date,

                "nearest_grid":
                    {
                        "latitude":
                            location.get(
                                "nearest_grid_latitude"
                            ),

                        "longitude":
                            location.get(
                                "nearest_grid_longitude"
                            ),
                    },

                "profile":
                    profile,

                "surface_inputs":
                    surface_inputs,

                "model_input":
                    {
                        "channels":
                            7,

                        "patch_shape":
                            [
                                7,
                                9,
                                9
                            ],
                    },
            }
        )


    except (
        TypeError,
        ValueError
    ) as e:

        return jsonify(
            {
                "success":
                    False,

                "type":
                    "INVALID_REQUEST",

                "error":
                    str(e),
            }
        ), 400


    except Exception as e:

        print()
        print(
            "PREDICTION ERROR:"
        )

        print(
            repr(e)
        )


        return jsonify(
            {
                "success":
                    False,

                "type":
                    "SERVER_ERROR",

                "error":
                    str(e),
            }
        ), 500


# ============================================================
# CROSS-SECTION BUILDER
# ============================================================

@lru_cache(
    maxsize=6
)
def build_cross_section(
    date_text,
    latitude
):

    latitude = float(
        latitude
    )


    # --------------------------------------------------------
    # FIND NEAREST MODEL LATITUDE
    # --------------------------------------------------------

    lat_index = int(
        np.argmin(
            np.abs(
                np.asarray(LATS)
                - latitude
            )
        )
    )


    # Keep enough space for a complete 9x9 patch

    lat_index = int(
        np.clip(
            lat_index,
            4,
            len(LATS) - 5
        )
    )


    section_latitude = float(
        LATS[
            lat_index
        ]
    )


    print()
    print(
        "=" * 70
    )

    print(
        "BUILDING THERMAL CROSS-SECTION"
    )

    print(
        f"Date: {date_text}"
    )

    print(
        f"Requested latitude: "
        f"{latitude:.3f}"
    )

    print(
        f"Model latitude: "
        f"{section_latitude:.3f}"
    )

    print(
        "Fetching online surface block..."
    )


    # --------------------------------------------------------
    # GET ONE FULL SURFACE DAY
    # --------------------------------------------------------

    _, surface_raw, dates = (
        load_surface_block(
            date_text,
            date_text
        )
    )


    if len(dates) == 0:

        raise ValueError(
            "No surface data returned "
            "for the selected date."
        )


    surface_day = np.asarray(
        surface_raw[0],
        dtype=np.float32
    )


    if surface_day.shape[0] != 7:

        raise ValueError(
            "Unexpected surface data shape: "
            + str(
                surface_day.shape
            )
        )


    # --------------------------------------------------------
    # LAND / OCEAN MASK
    # --------------------------------------------------------

    ocean_mask = get_ocean_mask()


    longitude_indices = (
        get_cross_section_longitudes()
    )


    patches = []

    valid_indices = []


    # --------------------------------------------------------
    # BUILD OCEAN PATCHES
    # --------------------------------------------------------

    for lon_index in longitude_indices:

        if not bool(
            ocean_mask[
                lat_index,
                lon_index
            ]
        ):

            continue


        patch = extract_surface_patch(
            surface_day,
            lat_index,
            lon_index
        )


        if patch.shape != (
            7,
            9,
            9
        ):

            continue


        patches.append(
            patch
        )


        valid_indices.append(
            lon_index
        )


    if not patches:

        raise ValueError(
            "No valid ocean points were found "
            "for this latitude."
        )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    patch_array = np.stack(
        patches,
        axis=0
    )


    patch_normalized = normalize_surface(
        patch_array
    )


    x = torch.from_numpy(
        patch_normalized.astype(
            np.float32
        )
    ).to(
        DEVICE
    )


    # --------------------------------------------------------
    # BATCH INFERENCE
    # --------------------------------------------------------

    if model is None:

        load_model()


    with torch.no_grad():

        predictions = model(
            x
        ).cpu().numpy()


    predictions = predictions.astype(
        np.float64
    )


    # predictions shape:
    #
    # [longitude, depth]
    #
    # Convert to:
    #
    # [depth, longitude]

    temperature_matrix = (
        predictions.T.tolist()
    )


    longitudes = [

        float(
            LONS[
                index
            ]
        )

        for index in valid_indices

    ]


    print(
        "Cross-section complete."
    )

    print(
        f"Ocean columns: "
        f"{len(longitudes)}"
    )

    print(
        "=" * 70
    )


    return {

        "success":
            True,

        "date":
            date_text,

        "requested_latitude":
            latitude,

        "section_latitude":
            section_latitude,

        "longitudes":
            longitudes,

        "depths":
            [
                float(x)
                for x in DEPTHS
            ],

        "temperature":
            temperature_matrix,

    }


# ============================================================
# CROSS-SECTION API
# ============================================================

@app.route(
    "/api/cross-section",
    methods=["POST"]
)
def cross_section():

    try:

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify(
                {
                    "success":
                        False,

                    "error":
                        "Request must contain JSON.",
                }
            ), 400


        latitude = float(
            data.get(
                "latitude"
            )
        )


        longitude = float(
            data.get(
                "longitude"
            )
        )


        selected_date = str(
            data.get(
                "date"
            )
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
                    "success":
                        False,

                    "type":
                        "OUT_OF_DOMAIN",

                    "message":
                        "Selected location is outside "
                        "the North Indian Ocean model domain.",
                }
            ), 400


        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        if not validate_date(
            selected_date
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "INVALID_DATE",

                    "message":
                        "Please select a date between "
                        "2023-01-01 and 2024-12-15.",
                }
            ), 400


        # ----------------------------------------------------
        # LAND / OCEAN CHECK
        # ----------------------------------------------------

        location = check_location(
            latitude,
            longitude
        )


        if not location[
            "is_ocean"
        ]:

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        location[
                            "status"
                        ],

                    "message":
                        location[
                            "message"
                        ],
                }
            ), 400


        # ----------------------------------------------------
        # BUILD CROSS-SECTION
        # ----------------------------------------------------

        result = build_cross_section(
            selected_date,
            round(
                latitude,
                4
            )
        )


        # ----------------------------------------------------
        # SELECTED LONGITUDE
        # ----------------------------------------------------

        result[
            "selected_longitude"
        ] = longitude


        result[
            "selected_grid_longitude"
        ] = location.get(
            "nearest_grid_longitude"
        )


        return jsonify(
            result
        )


    except (
        TypeError,
        ValueError
    ) as e:

        return jsonify(
            {
                "success":
                    False,

                "type":
                    "INVALID_REQUEST",

                "error":
                    str(e),
            }
        ), 400


    except Exception as e:

        print()
        print(
            "CROSS-SECTION ERROR:"
        )

        print(
            repr(e)
        )


        return jsonify(
            {
                "success":
                    False,

                "type":
                    "SERVER_ERROR",

                "error":
                    str(e),
            }
        ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    load_model()


    print()
    print(
        "=" * 70
    )


    print(
        "OCEANEMBED WEB APPLICATION"
    )


    print(
        "=" * 70
    )


    print(
        "Domain: 5°N to 30°N"
    )


    print(
        "Longitude: 45°E to 105°E"
    )


    print(
        "Training year: 2023"
    )


    print(
        "Validation year: 2024"
    )


    print(
        "Surface data: ONLINE"
    )


    print(
        "Thermal cross-section: ENABLED"
    )


    print(
        "Depths:"
    )


    print(
        [
            float(x)
            for x in DEPTHS
        ]
    )


    print()
    print(
        "Server:"
    )


    print(
        "http://127.0.0.1:5000"
    )


    print(
        "=" * 70
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )