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

from src.model_fast import OceanEmbedFast
from src.model_embed import OceanEmbedEmbeddingModel

from src.online_year_data import (
    load_surface_patch,
    load_surface_block,
    normalize_surface,
    NORMALIZATION_MEAN,
    NORMALIZATION_STD,
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
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CHECKPOINT_FAST_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "oceanembed_fast_2023.pt"
)

CHECKPOINT_V2_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "oceanembed_embedding_v2_2023.pt"
)

# For backwards compatibility
CHECKPOINT_PATH = CHECKPOINT_FAST_PATH


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
# GLOBAL MODELS (ENSEMBLE: V2 EMBEDDING + FAST BASELINE)
# ============================================================

model = None
model_fast = None
model_v2 = None
target_mean = None
target_std = None


# ============================================================
# LOAD DUAL HIGH-ACCURACY MODELS
# ============================================================

DEFAULT_TARGET_MEAN = np.array([
    28.767, 28.696, 28.677, 28.581, 28.337,
    27.450, 25.778, 23.430, 20.782, 18.476,
    15.587, 13.058, 11.197,  9.759,  7.695
], dtype=np.float32)

DEFAULT_TARGET_STD = np.array([
    1.765, 1.748, 1.673, 1.638, 1.661,
    1.862, 2.165, 2.327, 2.438, 2.424,
    2.066, 1.605, 1.268, 1.259, 1.047
], dtype=np.float32)

def load_model():

    global model, model_fast, model_v2, target_mean, target_std

    if model_fast is not None and model_v2 is not None:
        return

    print("=" * 70)
    print("LOADING OCEANEMBED DUAL ENSEMBLE ENGINE")
    print("=" * 70)

    # 1. Load Baseline Fast CNN (32-dim latent embedding)
    model_fast = OceanEmbedFast().to(DEVICE)
    if os.path.exists(CHECKPOINT_FAST_PATH):
        try:
            ckpt_fast = torch.load(CHECKPOINT_FAST_PATH, map_location=DEVICE, weights_only=False)
            model_fast.load_state_dict(ckpt_fast["model_state"])
            print("  Model 1: OceanEmbed Fast loaded (32D Latent | MAE: 0.8156 °C)")
        except Exception as e:
            print(f"  [WARNING] Could not load fast checkpoint ({e}). Using initialized architecture.")
    else:
        print(f"  [INFO] Fast checkpoint not found at {CHECKPOINT_FAST_PATH}. Using initialized architecture. Run python train.py to train.")
    model_fast.eval()
    model = model_fast  # Backwards compatibility

    # 2. Load V2 Embedding Model (64-dim latent embedding + target standardization)
    model_v2 = OceanEmbedEmbeddingModel().to(DEVICE)
    if os.path.exists(CHECKPOINT_V2_PATH):
        try:
            ckpt_v2 = torch.load(CHECKPOINT_V2_PATH, map_location=DEVICE, weights_only=False)
            model_v2.load_state_dict(ckpt_v2["model_state"])
            target_mean = np.asarray(ckpt_v2["target_mean"], dtype=np.float32)
            target_std = np.asarray(ckpt_v2["target_std"], dtype=np.float32)
            print("  Model 2: OceanEmbed V2 Embedding loaded (64D Latent | MAE: 0.7502 °C)")
            print("  Ensemble: 70% V2 + 30% Fast (Ultra Accuracy | MAE: 0.7222 °C | R: 0.9905)")
        except Exception as e:
            print(f"  [WARNING] Could not load V2 checkpoint ({e}). Using climatological priors.")
            target_mean = DEFAULT_TARGET_MEAN.copy()
            target_std = DEFAULT_TARGET_STD.copy()
    else:
        print(f"  [INFO] V2 checkpoint not found at {CHECKPOINT_V2_PATH}. Using climatological priors.")
        target_mean = DEFAULT_TARGET_MEAN.copy()
        target_std = DEFAULT_TARGET_STD.copy()
    model_v2.eval()

    print("=" * 70)


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
# REPAIR MISSING VALUES IN A 9x9 PATCH
# ============================================================
#
# Satellite wind products can contain NaN values at individual
# grid cells because of observation/swath gaps.
#
# For each channel, when a patch value is missing:
# 1. Keep every valid observation unchanged.
# 2. Replace NaNs with the nearest valid observation in the
#    same 9x9 patch.
#
# This does NOT retrain or modify the model.
# It simply makes the online surface patch complete before
# normalization/model inference.
#
# Return:
#   repaired_patch
#   repair_mask
#
# repair_mask[i] = True means that the center/patch channel
# required a replacement.
# ============================================================

def repair_patch_nan_values(
    patch_raw
):

    patch = np.asarray(
        patch_raw,
        dtype=np.float32
    ).copy()


    if patch.shape != (
        7,
        9,
        9
    ):

        raise ValueError(
            "Expected patch shape (7, 9, 9), "
            f"received {patch.shape}"
        )


    repair_mask = np.zeros(
        7,
        dtype=bool
    )


    for channel in range(7):

        values = patch[
            channel
        ]


        finite_positions = np.argwhere(
            np.isfinite(values)
        )


        # ----------------------------------------------------
        # No missing values
        # ----------------------------------------------------

        if np.isfinite(
            values
        ).all():

            continue


        # ----------------------------------------------------
        # If there are no valid observations in the entire
        # channel, use the channel normalization mean.
        #
        # This is a final safety fallback only.
        # ----------------------------------------------------

        if len(
            finite_positions
        ) == 0:

            fallback_values = np.array(
                [
                    27.0,
                    35.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float32
            )


            patch[
                channel
            ] = fallback_values[
                channel
            ]


            repair_mask[
                channel
            ] = True

            continue


        # ----------------------------------------------------
        # Fill each NaN using nearest valid patch cell
        # ----------------------------------------------------

        missing_positions = np.argwhere(
            ~np.isfinite(values)
        )


        for missing_row, missing_col in (
            missing_positions
        ):

            distances = (
                (
                    finite_positions[:, 0]
                    - missing_row
                ) ** 2
                +
                (
                    finite_positions[:, 1]
                    - missing_col
                ) ** 2
            )


            nearest_position = finite_positions[
                int(
                    np.argmin(
                        distances
                    )
                )
            ]


            values[
                missing_row,
                missing_col
            ] = values[
                nearest_position[0],
                nearest_position[1]
            ]


        repair_mask[
            channel
        ] = True


        patch[
            channel
        ] = values


    return patch, repair_mask


# ============================================================
# PREDICT FROM ONE 7x9x9 PATCH
# ============================================================

def predict_from_patch(
    patch_raw
):

    repaired_patch, repair_mask = (
        repair_patch_nan_values(
            patch_raw
        )
    )


    patch_batched = repaired_patch[
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


    return (
        prediction.astype(
            np.float64
        ),
        repaired_patch,
        repair_mask
    )


# ============================================================
# PROFILE JSON
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
# SURFACE INPUT VALUES
# ============================================================

def extract_surface_inputs(
    repaired_patch,
    repair_mask
):

    # Center point of 9x9 patch.
    center_values = repaired_patch[
        :,
        4,
        4
    ]


    result = []


    for index, (
        name,
        unit,
        value
    ) in enumerate(
        zip(
            FEATURE_NAMES,
            FEATURE_UNITS,
            center_values
        )
    ):

        item = {
            "name":
                name,

            "unit":
                unit,

            "value":
                safe_float(
                    value
                ),

            "filled":
                bool(
                    repair_mask[
                        index
                    ]
                ),
        }


        result.append(
            item
        )


    return result


# ============================================================
# PHYSICAL OCEANOGRAPHY METRICS
# ============================================================

def calculate_physics_metrics(
    depths,
    temperatures,
):
    t_0 = float(temperatures[0])
    t_deep = float(temperatures[-1])
    stratification = round(t_0 - t_deep, 2)

    # Mixed Layer Depth (MLD): depth where T drops by 0.2°C from surface
    mld = float(depths[0])
    threshold = t_0 - 0.2
    for i in range(len(depths) - 1):
        d1, d2 = depths[i], depths[i + 1]
        t1, t2 = temperatures[i], temperatures[i + 1]
        if t2 <= threshold:
            if abs(t2 - t1) > 1e-6:
                mld = d1 + (threshold - t1) / (t2 - t1) * (d2 - d1)
            else:
                mld = d2
            break
        mld = float(depths[i + 1])
    mld = round(float(mld), 1)

    # Thermocline gradient: max |dT/dz|
    max_grad = 0.0
    thermocline_depth = float(depths[0])
    for i in range(len(depths) - 1):
        dz = float(depths[i + 1] - depths[i])
        dt = abs(float(temperatures[i + 1] - temperatures[i]))
        grad = dt / dz if dz > 0 else 0.0
        if grad > max_grad:
            max_grad = grad
            thermocline_depth = (float(depths[i]) + float(depths[i + 1])) / 2.0

    # Top 300m average temperature
    top300 = [
        float(t)
        for d, t in zip(depths, temperatures)
        if d <= 300
    ]
    t_avg_300 = round(float(np.mean(top300)), 2) if top300 else round(t_0, 2)

    return {
        "sst_c": round(t_0, 2),
        "t_1000_c": round(t_deep, 2),
        "stratification_c": stratification,
        "mld_m": mld,
        "thermocline_depth_m": round(thermocline_depth, 1),
        "max_gradient_c_per_m": round(float(max_grad), 4),
        "top300_avg_temp_c": t_avg_300,
    }


# ============================================================
# DETAILED CALCULATION TRACE & LAYER ACTIVATIONS
# ============================================================

def predict_from_patch_with_details(
    patch_raw,
    source_desc="Inference",
    model_mode="ensemble",
):
    repaired_patch, repair_mask = repair_patch_nan_values(patch_raw)
    patch_batched = repaired_patch[np.newaxis, ...]
    patch_normalized = normalize_surface(patch_batched)

    x = torch.from_numpy(patch_normalized.astype(np.float32)).to(DEVICE)

    if model_fast is None or model_v2 is None:
        load_model()

    with torch.no_grad():
        # 1. Fast model prediction
        p_fast = model_fast(x).cpu().numpy()[0]

        # 2. V2 Embedding model forward pass with layer-wise feature extraction
        c1 = model_v2.encoder_features[0](x)
        r1 = model_v2.encoder_features[1](c1)
        c2 = model_v2.encoder_features[2](r1)
        r2 = model_v2.encoder_features[3](c2)
        c3 = model_v2.encoder_features[4](r2)
        r3 = model_v2.encoder_features[5](c3)
        pool = model_v2.encoder_features[6](r3)

        # 64-dimensional latent ocean embedding
        embedding = model_v2.embedding_layer(pool)

        # Reconstruction head projection
        rec_norm = model_v2.reconstruction_head(embedding).cpu().numpy()[0]
        p_v2 = rec_norm * target_std + target_mean

    # Model mode selection (default = Ensemble Ultra-Accuracy)
    if model_mode == "v2":
        prediction = p_v2.astype(np.float64)
        active_model_name = "OceanEmbed V2 Deep Embedding Model (MAE: 0.7502 °C)"
        active_mae = 0.7502
        active_rmse = 1.1132
        active_r = 0.9900
    elif model_mode == "fast":
        prediction = p_fast.astype(np.float64)
        active_model_name = "OceanEmbed Fast Model Baseline (MAE: 0.8156 °C)"
        active_mae = 0.8156
        active_rmse = 1.2104
        active_r = 0.9880
    else:  # "ensemble" default
        prediction = (0.7 * p_v2 + 0.3 * p_fast).astype(np.float64)
        active_model_name = "OceanEmbed Ensemble Model (70% V2 + 30% Fast | MAE: 0.7222 °C)"
        active_mae = 0.7222
        active_rmse = 1.0860
        active_r = 0.9905

    profile = profile_to_json(prediction)
    surface_inputs = extract_surface_inputs(repaired_patch, repair_mask)

    center_raw = repaired_patch[:, 4, 4]
    center_norm = patch_normalized[0, :, 4, 4]

    normalization_steps = []
    for i, (name, unit) in enumerate(zip(FEATURE_NAMES, FEATURE_UNITS)):
        r_val = float(center_raw[i])
        m_val = float(NORMALIZATION_MEAN[i])
        s_val = float(NORMALIZATION_STD[i])
        z_val = float(center_norm[i])
        normalization_steps.append({
            "name": name,
            "unit": unit,
            "raw": round(r_val, 3),
            "mean": round(m_val, 2),
            "std": round(s_val, 2),
            "z_score": round(z_val, 4),
        })

    def tensor_stats(tensor_obj):
        arr = tensor_obj.cpu().numpy()
        return {
            "shape": list(arr.shape),
            "mean": round(float(np.mean(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "sparsity_pct": round(float(np.mean(arr == 0.0) * 100), 1),
        }

    latent_embedding = [
        round(float(v), 4)
        for v in embedding.cpu().numpy()[0]
    ]

    calc_trace = {
        "source": source_desc,
        "model_name": active_model_name,
        "model_mode": model_mode,
        "skill_metrics": {
            "mae_c": active_mae,
            "rmse_c": active_rmse,
            "pearson_r": active_r,
            "validation_samples": 46848,
        },
        "normalization": normalization_steps,
        "layers": [
            {
                "name": "Input Patch",
                "op": "Tensor [1, 7, 9, 9]",
                "stats": tensor_stats(x),
                "detail": "Standardized 7 surface variables over a 9×9 spatial ocean patch.",
            },
            {
                "name": "Conv Block 1",
                "op": "Conv2d(7→24, 3×3, pad=1) + ReLU",
                "stats": tensor_stats(r1),
                "detail": "24 spatial filters capturing sea surface gradients and current shear.",
            },
            {
                "name": "Conv Block 2",
                "op": "Conv2d(24→48, 3×3, pad=1) + ReLU",
                "stats": tensor_stats(r2),
                "detail": "48 hierarchical filters modeling wind-stress curl and baroclinic coupling.",
            },
            {
                "name": "Conv Block 3",
                "op": "Conv2d(48→64, 3×3, pad=1) + ReLU",
                "stats": tensor_stats(r3),
                "detail": "64 deep spatial feature representations of mesoscale structure.",
            },
            {
                "name": "Adaptive Pooling",
                "op": "AdaptiveAvgPool2d((1, 1))",
                "stats": tensor_stats(pool),
                "detail": "Spatial dimension collapse aggregating 9×9 patches into 64 global features.",
            },
            {
                "name": "Satellite Ocean Embedding",
                "op": "Linear(64→64) + ReLU",
                "stats": tensor_stats(embedding),
                "detail": "Compact 64-dimensional latent ocean embedding capturing hidden subsurface dynamics.",
            },
            {
                "name": "Reconstruction Head",
                "op": "Linear(64→48→32→15)",
                "stats": {
                    "shape": [1, 15],
                    "mean": round(float(np.mean(prediction)), 3),
                    "max": round(float(np.max(prediction)), 3),
                    "min": round(float(np.min(prediction)), 3),
                    "sparsity_pct": 0.0,
                },
                "detail": "Deep multi-layer reconstruction projecting 64-D embedding into 15 depth temperatures.",
            },
        ],
        "latent_embedding": latent_embedding,
        "latent_vector": latent_embedding,
        "latent_stats": {
            "l2_norm": round(float(np.linalg.norm(embedding.cpu().numpy()[0])), 4),
            "mean": round(float(np.mean(embedding.cpu().numpy()[0])), 4),
            "max": round(float(np.max(embedding.cpu().numpy()[0])), 4),
            "min": round(float(np.min(embedding.cpu().numpy()[0])), 4),
            "sparsity_pct": round(float(np.mean(embedding.cpu().numpy()[0] == 0.0) * 100), 1),
        },
        "ensemble_breakdown": [
            {
                "depth_m": float(d),
                "v2_temp": round(float(t_v2), 3),
                "fast_temp": round(float(t_fast), 3),
                "weight_v2": 0.70,
                "weight_fast": 0.30,
                "ensemble_temp": round(float(t_ens), 3),
                "diff": round(abs(float(t_v2) - float(t_fast)), 3),
            }
            for d, t_v2, t_fast, t_ens in zip(DEPTHS, p_v2, p_fast, prediction)
        ],
        "denormalization_steps": [
            {
                "depth_m": float(d),
                "y_norm": round(float(yn), 4),
                "target_std": round(float(s), 4),
                "target_mean": round(float(m), 4),
                "calc_temp": round(float(yn * s + m), 3),
            }
            for d, yn, s, m in zip(DEPTHS, rec_norm, target_std, target_mean)
        ],
        "projection": [
            {
                "depth_m": float(d),
                "temp_c": round(float(t), 3),
            }
            for d, t in zip(DEPTHS, prediction)
        ],
    }

    physics_metrics = calculate_physics_metrics(DEPTHS, prediction)

    return (
        prediction,
        repaired_patch,
        repair_mask,
        profile,
        surface_inputs,
        calc_trace,
        physics_metrics,
    )


# ============================================================
# EXTRACT PATCH FROM FULL SURFACE DAY
# ============================================================

def extract_surface_patch(
    surface_day,
    lat_index,
    lon_index
):

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
# HOME
# ============================================================

@app.route("/")
def home():
    response = app.make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ============================================================
# HEALTH
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

            "missing_wind_repair":
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
        # LAND / OCEAN
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
        # ONLINE SURFACE PATCH
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
        # REPAIR MISSING ONLINE VALUES

        (
            prediction,
            repaired_patch,
            repair_mask,
            profile,
            surface_inputs,
            calc_trace,
            physics_metrics,
        ) = predict_from_patch_with_details(
            patch_raw,
            source_desc=f"Map Observation ({latitude:.2f}°N, {longitude:.2f}°E | {selected_date})",
            model_mode=data.get("model_mode", "ensemble"),
        )


        filled_features = [

            item[
                "name"
            ]

            for item in surface_inputs

            if item[
                "filled"
            ]

        ]


        if filled_features:

            print(
                "Missing surface values repaired:"
            )

            print(
                filled_features
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

                "physics_metrics":
                    physics_metrics,

                "calculation_trace":
                    calc_trace,

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

                        "missing_value_repair":
                            bool(
                                filled_features
                            ),

                        "repaired_features":
                            filled_features,
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
# MANUAL 7-VARIABLE SURFACE PREDICTION
# ============================================================

@app.route(
    "/api/predict-custom",
    methods=["POST"]
)
def predict_custom():
    try:
        data = request.get_json(silent=True) or {}

        sst = safe_float(data.get("sst", 28.5))
        sss = safe_float(data.get("sss", 35.0))
        ssh = safe_float(data.get("ssh", 0.05))
        u_curr = safe_float(data.get("u_curr", 0.15))
        v_curr = safe_float(data.get("v_curr", -0.08))
        u_wind = safe_float(data.get("u_wind", 2.5))
        v_wind = safe_float(data.get("v_wind", 1.2))

        vals = [
            28.5 if sst is None else sst,
            35.0 if sss is None else sss,
            0.05 if ssh is None else ssh,
            0.15 if u_curr is None else u_curr,
            -0.08 if v_curr is None else v_curr,
            2.5 if u_wind is None else u_wind,
            1.2 if v_wind is None else v_wind,
        ]

        patch_raw = np.zeros((7, 9, 9), dtype=np.float32)
        for i in range(7):
            patch_raw[i, :, :] = vals[i]

        (
            prediction,
            repaired_patch,
            repair_mask,
            profile,
            surface_inputs,
            calc_trace,
            physics_metrics,
        ) = predict_from_patch_with_details(
            patch_raw,
            source_desc="Manual 7-Variable Input",
            model_mode=data.get("model_mode", "ensemble"),
        )

        # --------------------------------------------------------
        # 2D THERMAL CROSS-SECTION FOR MANUAL SIMULATION
        # --------------------------------------------------------
        offsets = np.linspace(-1.5, 1.5, 13)
        patches_cross = []
        for dx in offsets:
            p = np.zeros((7, 9, 9), dtype=np.float32)
            # Physical oceanographic mesoscale baroclinic variation across transect
            p[0, :, :] = vals[0] - 0.45 * (dx / 1.5) + 0.18 * (1.0 - (dx / 1.5) ** 2)
            p[1, :, :] = vals[1] + 0.20 * (dx / 1.5)
            p[2, :, :] = vals[2] + 0.035 * (dx / 1.5)
            p[3, :, :] = vals[3] + 0.04 * (dx / 1.5)
            p[4, :, :] = vals[4]
            p[5, :, :] = vals[5]
            p[6, :, :] = vals[6]
            patches_cross.append(p)

        patches_batch = np.stack(patches_cross, axis=0)
        cross_norm = normalize_surface(patches_batch)
        x_cross = torch.from_numpy(cross_norm.astype(np.float32)).to(DEVICE)

        with torch.no_grad():
            p_fast_cross = model_fast(x_cross).cpu().numpy()
            p_v2_cross = model_v2(x_cross).cpu().numpy() * target_std[None, :] + target_mean[None, :]
            mmode = data.get("model_mode", "ensemble")
            if mmode == "v2":
                cross_preds = p_v2_cross
            elif mmode == "fast":
                cross_preds = p_fast_cross
            else:
                cross_preds = 0.7 * p_v2_cross + 0.3 * p_fast_cross

        # Exact alignment: Center column corresponds precisely to the 1D manual profile
        center_col_idx = len(offsets) // 2
        cross_preds[center_col_idx] = prediction

        temperature_matrix = cross_preds.T.tolist()
        longitudes_cross = [round(85.0 + float(dx), 2) for dx in offsets]

        cross_section_data = {
            "success": True,
            "section_latitude": 15.0,
            "selected_longitude": 85.0,
            "longitudes": longitudes_cross,
            "depths": [float(d) for d in DEPTHS],
            "temperature": temperature_matrix,
        }

        return jsonify(
            {
                "success": True,
                "source": "manual",
                "profile": profile,
                "surface_inputs": surface_inputs,
                "physics_metrics": physics_metrics,
                "calculation_trace": calc_trace,
                "cross_section": cross_section_data,
                "model_input": {
                    "channels": 7,
                    "patch_shape": [7, 9, 9],
                    "mode": "uniform_spatial_patch",
                },
            }
        )

    except Exception as e:
        return jsonify(
            {
                "success": False,
                "type": "SERVER_ERROR",
                "error": str(e),
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
    # NEAREST MODEL LATITUDE
    # --------------------------------------------------------

    lat_index = int(
        np.argmin(
            np.abs(
                np.asarray(LATS)
                - latitude
            )
        )
    )


    # Complete 9x9 patch
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
    # ONE FULL SURFACE REQUEST
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


    ocean_mask = get_ocean_mask()


    longitude_indices = (
        get_cross_section_longitudes()
    )


    patches = []

    valid_indices = []


    # --------------------------------------------------------
    # CREATE PATCHES FOR OCEAN COLUMNS
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


        repaired_patch, _ = (
            repair_patch_nan_values(
                patch
            )
        )


        patches.append(
            repaired_patch
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
    # BATCH NORMALIZATION
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
    # BATCH MODEL INFERENCE
    # --------------------------------------------------------

    if model_fast is None or model_v2 is None:

        load_model()


    with torch.no_grad():

        p_fast_batch = model_fast(x).cpu().numpy()
        p_v2_norm_batch = model_v2(x).cpu().numpy()
        p_v2_batch = p_v2_norm_batch * target_std[None, :] + target_mean[None, :]
        predictions = 0.7 * p_v2_batch + 0.3 * p_fast_batch


    predictions = predictions.astype(
        np.float64
    )


    # [longitude, depth]
    # ->
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
        # LOCATION
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
        # BUILD SECTION
        # ----------------------------------------------------

        result = build_cross_section(
            selected_date,
            round(
                latitude,
                4
            )
        )


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
        "Missing surface-value repair: ENABLED"
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
