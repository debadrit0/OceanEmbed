import os
import sys

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request


# ============================================================
# PATH SETUP
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

if SRC_DIR not in sys.path:
    sys.path.insert(
        0,
        SRC_DIR
    )


# ============================================================
# IMPORT OCEANEMBED COMPONENTS
# ============================================================

from model_fast import OceanEmbedFast

from online_year_data import (
    load_surface_patch,
    normalize_surface,
)

from ocean_mask import check_location


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
# LOAD MODEL ONCE
# ============================================================

model = None


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
        "OceanEmbed model loaded."
    )


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
            "status": "ok",
            "model_loaded": model is not None,
            "domain": {
                "lat_min": 5,
                "lat_max": 30,
                "lon_min": 45,
                "lon_max": 105,
            },
            "depths": [
                float(x)
                for x in DEPTHS
            ]
        }
    )


# ============================================================
# PREDICTION API
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
                    "success": False,
                    "error": "Request must contain JSON."
                }
            ), 400


        # ----------------------------------------------------
        # INPUTS
        # ----------------------------------------------------

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
        # DOMAIN VALIDATION
        # ----------------------------------------------------

        if (
            latitude < 5
            or latitude > 30
            or longitude < 45
            or longitude > 105
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "OUT_OF_DOMAIN",
                    "message":
                        "Selected location is outside the "
                        "North Indian Ocean model domain."
                }
            ), 400


        # ----------------------------------------------------
        # DATE VALIDATION
        #
        # We restrict the demo to dates for which all of the
        # currently configured historical input sources can
        # be used without fabricating the entire surface state.
        # ----------------------------------------------------

        if (
            selected_date < "2023-01-01"
            or selected_date > "2024-12-15"
        ):

            return jsonify(
                {
                    "success": False,
                    "type": "INVALID_DATE",
                    "message":
                        "Please select a date between "
                        "2023-01-01 and 2024-12-15."
                }
            ), 400


        # ----------------------------------------------------
        # LAND / OCEAN CHECK
        # ----------------------------------------------------

        location = check_location(
            latitude,
            longitude
        )


        if not location["is_ocean"]:

            return jsonify(
                {
                    "success": False,
                    "type": "LAND",
                    "message":
                        "Selected location is a land mass. "
                        "Please select a location over the ocean."
                }
            ), 400


        # ----------------------------------------------------
        # ONLINE SURFACE DATA
        # ----------------------------------------------------

        print()
        print(
            "=========================================="
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
        # NORMALIZE
        # ----------------------------------------------------

        patch_raw_batched = patch_raw[
            np.newaxis,
            ...
        ]


        patch_norm = normalize_surface(
            patch_raw_batched
        )


        # ----------------------------------------------------
        # MODEL INPUT
        # ----------------------------------------------------

        x = torch.from_numpy(
            patch_norm.astype(
                np.float32
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = model(
                x.to(DEVICE)
            ).cpu().numpy()[0]


        prediction = prediction.astype(
            np.float64
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        profile = []

        for depth, temperature in zip(
            DEPTHS,
            prediction
        ):

            profile.append(
                {
                    "depth_m":
                        float(depth),

                    "temperature_c":
                        round(
                            float(
                                temperature
                            ),
                            3
                        )
                }
            )


        print(
            "Prediction complete."
        )

        print(
            "=========================================="
        )


        return jsonify(
            {
                "success": True,

                "location": {
                    "latitude":
                        latitude,

                    "longitude":
                        longitude
                },

                "date":
                    selected_date,

                "nearest_grid": {
                    "latitude":
                        location.get(
                            "nearest_lat"
                        ),

                    "longitude":
                        location.get(
                            "nearest_lon"
                        )
                },

                "profile":
                    profile
            }
        )


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
                "success": False,
                "type": "SERVER_ERROR",
                "error": str(e)
            }
        ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    load_model()

    print()
    print("=" * 70)

    print(
        "OCEANEMBED WEB APPLICATION"
    )

    print("=" * 70)

    print(
        "Domain: "
        "5-30 N, 45-105 E"
    )

    print(
        "Training: 2023"
    )

    print(
        "Validation: 2024"
    )

    print(
        "Surface data: ONLINE"
    )

    print(
        "Server: http://127.0.0.1:5000"
    )

    print("=" * 70)

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )