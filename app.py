import os

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request


# ============================================================
# OCEANEMBED IMPORTS
# ============================================================

from src.model_fast import OceanEmbedFast

from src.online_year_data import (
    load_surface_patch,
    normalize_surface
)

from src.ocean_mask import check_location


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    template_folder="templates"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
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
        1000
    ],
    dtype=np.float32
)


DEVICE = torch.device(
    "cpu"
)


# ============================================================
# MODEL
# ============================================================

model = None


def load_model():

    global model

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "OceanEmbed trained model was not found.\n\n"
            "Expected file:\n"
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
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
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

            "model_loaded":
                model is not None,

            "domain":
                {
                    "latitude_min": 5,
                    "latitude_max": 30,
                    "longitude_min": 45,
                    "longitude_max": 105
                },

            "depths":
                [
                    float(depth)
                    for depth in DEPTHS
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

        # ----------------------------------------------------
        # READ JSON
        # ----------------------------------------------------

        data = request.get_json(
            silent=True
        )


        if data is None:

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "INVALID_REQUEST",

                    "message":
                        "Request must contain JSON."
                }
            ), 400


        # ----------------------------------------------------
        # CHECK REQUIRED INPUTS
        # ----------------------------------------------------

        if (
            "latitude" not in data
            or
            "longitude" not in data
            or
            "date" not in data
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "MISSING_INPUT",

                    "message":
                        "Latitude, longitude and date are required."
                }
            ), 400


        latitude = float(
            data["latitude"]
        )

        longitude = float(
            data["longitude"]
        )

        selected_date = str(
            data["date"]
        )


        # ----------------------------------------------------
        # DOMAIN CHECK
        # ----------------------------------------------------

        if (
            latitude < 5
            or
            latitude > 30
            or
            longitude < 45
            or
            longitude > 105
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "OUT_OF_DOMAIN",

                    "message":
                        "Selected location is outside the "
                        "North Indian Ocean model domain."
                }
            ), 400


        # ----------------------------------------------------
        # DATE CHECK
        # ----------------------------------------------------

        if (
            selected_date < "2023-01-01"
            or
            selected_date > "2024-12-15"
        ):

            return jsonify(
                {
                    "success":
                        False,

                    "type":
                        "INVALID_DATE",

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
                    "success":
                        False,

                    "type":
                        "LAND",

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
            "=" * 70
        )

        print(
            "NEW OCEANEMBED REQUEST"
        )

        print(
            f"Latitude : {latitude:.4f}"
        )

        print(
            f"Longitude: {longitude:.4f}"
        )

        print(
            f"Date     : {selected_date}"
        )

        print(
            "Location status: OCEAN"
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
        # INPUT SHAPE CHECK
        # ----------------------------------------------------

        if patch_raw is None:

            raise RuntimeError(
                "Online surface data could not be retrieved."
            )


        if patch_raw.shape != (
            7,
            9,
            9
        ):

            raise RuntimeError(
                "Unexpected surface input shape: "
                f"{patch_raw.shape}. "
                "Expected (7, 9, 9)."
            )


        # ----------------------------------------------------
        # BATCH DIMENSION
        # ----------------------------------------------------

        patch_batched = patch_raw[
            np.newaxis,
            ...
        ]


        # ----------------------------------------------------
        # NORMALIZATION
        # ----------------------------------------------------

        patch_normalized = normalize_surface(
            patch_batched
        )


        # ----------------------------------------------------
        # PYTORCH INPUT
        # ----------------------------------------------------

        input_tensor = torch.from_numpy(
            patch_normalized.astype(
                np.float32
            )
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = model(
                input_tensor.to(
                    DEVICE
                )
            )


        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )


        prediction = prediction.astype(
            np.float64
        )


        # ----------------------------------------------------
        # OUTPUT CHECK
        # ----------------------------------------------------

        if prediction.shape != (
            15,
        ):

            raise RuntimeError(
                "Unexpected model output shape: "
                f"{prediction.shape}. "
                "Expected (15,)."
            )


        # ----------------------------------------------------
        # BUILD PROFILE
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


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        response = {

            "success":
                True,

            "location":
                {
                    "latitude":
                        latitude,

                    "longitude":
                        longitude
                },

            "date":
                selected_date,

            "nearest_grid":
                {
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


        print(
            "Prediction completed successfully."
        )

        print(
            "=" * 70
        )


        return jsonify(
            response
        )


    except Exception as error:

        print()
        print(
            "=" * 70
        )

        print(
            "OCEANEMBED SERVER ERROR"
        )

        print(
            repr(error)
        )

        print(
            "=" * 70
        )


        return jsonify(
            {
                "success":
                    False,

                "type":
                    "SERVER_ERROR",

                "message":
                    str(error)
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
        "Domain: "
        "5°N to 30°N"
    )

    print(
        "Longitude: "
        "45°E to 105°E"
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
        "Depths:"
    )

    print(
        DEPTHS.tolist()
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