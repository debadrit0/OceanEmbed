import os
import random
from datetime import date, timedelta

import numpy as np
import torch
import xarray as xr

from .model_embed import OceanEmbedEmbeddingModel

from .online_year_data import (
    load_surface_block,
    extract_patch,
    TARGET_LAT,
    TARGET_LON,
    HALF_PATCH,
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_YEAR = 2024

BLOCK_DAYS = 30

SAMPLES_PER_DAY = 128

GLORYS_PATH = os.path.join(
    "data",
    "processed",
    "glorys_full.nc"
)

CHECKPOINT_PATH = os.path.join(
    "checkpoints",
    "oceanembed_embedding_2023.pt"
)

RESULT_DIR = os.path.join(
    "results"
)

RESULT_PATH = os.path.join(
    RESULT_DIR,
    "validation_embedding_2024.npz"
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

SEED = 123


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cpu"
)

torch.set_num_threads(
    max(
        1,
        min(
            8,
            os.cpu_count() or 1
        )
    )
)

print(
    "Device:",
    DEVICE
)

print(
    "CPU threads:",
    torch.get_num_threads()
)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(
    GLORYS_PATH
):

    raise FileNotFoundError(
        "GLORYS file not found:\n"
        + GLORYS_PATH
    )


if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        "Embedding model checkpoint not found:\n"
        + CHECKPOINT_PATH
        + "\n\n"
        "Run the embedding-model training first."
    )


os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING OCEANEMBED EMBEDDING MODEL")
print("=" * 70)


model = (
    OceanEmbedEmbeddingModel()
    .to(DEVICE)
)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state"
    ]
)


model.eval()


print(
    "Model loaded:"
)

print(
    CHECKPOINT_PATH
)

print(
    "Embedding dimension:",
    model.EMBEDDING_DIM
)


# ============================================================
# LOAD GLORYS
# ============================================================

print()
print("=" * 70)
print("OPENING GLORYS")
print("=" * 70)


glorys = xr.open_dataset(
    GLORYS_PATH
)


if "temperature" not in glorys.data_vars:

    raise RuntimeError(
        "temperature variable not found "
        "in GLORYS.\n"
        f"Available variables: "
        f"{list(glorys.data_vars)}"
    )


temperature = glorys[
    "temperature"
]


print(
    "GLORYS dimensions:",
    temperature.dims
)

print(
    "GLORYS shape:",
    temperature.shape
)

print(
    "Depths:",
    temperature.depth.values
)


# ============================================================
# VERIFY REQUIRED DEPTHS
# ============================================================

available_depths = np.asarray(
    temperature.depth.values,
    dtype=np.float32
)


for depth in DEPTHS:

    if not np.any(
        np.isclose(
            available_depths,
            depth,
            atol=0.01
        )
    ):

        raise RuntimeError(
            f"Required depth {depth} m "
            "is not present in GLORYS."
        )


# ============================================================
# SELECT 2024
# ============================================================

validation_temperature = (
    temperature.sel(
        time=slice(
            "2024-01-01",
            "2024-12-31"
        )
    )
)


if len(
    validation_temperature.time
) == 0:

    raise RuntimeError(
        "No GLORYS data found for 2024."
    )


validation_dates = [

    date(
        VALIDATION_YEAR,
        1,
        1
    )
    + timedelta(
        days=i
    )

    for i in range(
        len(
            validation_temperature.time
        )
    )

]


print()

print(
    "Validation period:"
)

print(
    str(
        validation_temperature
        .time
        .values[0]
    )
)

print(
    "to"
)

print(
    str(
        validation_temperature
        .time
        .values[-1]
    )
)

print(
    "Validation days:",
    len(
        validation_dates
    )
)


# ============================================================
# CREATE BLOCKS
# ============================================================

blocks = []

start_index = 0


while (
    start_index
    < len(
        validation_dates
    )
):

    end_index = min(
        start_index + BLOCK_DAYS,
        len(validation_dates)
    )


    blocks.append(
        (
            start_index,
            end_index
        )
    )


    start_index = end_index


print(
    "Validation blocks:",
    len(blocks)
)


# ============================================================
# RESULT STORAGE
# ============================================================

all_predictions = []

all_targets = []


# ============================================================
# VALIDATION LOOP
# ============================================================

for block_number, (
    block_start,
    block_end
) in enumerate(
    blocks,
    start=1
):

    block_dates = validation_dates[
        block_start:block_end
    ]


    start_text = block_dates[
        0
    ].strftime(
        "%Y-%m-%d"
    )


    end_text = block_dates[
        -1
    ].strftime(
        "%Y-%m-%d"
    )


    print()
    print("=" * 70)

    print(
        f"EMBEDDING MODEL VALIDATION "
        f"BLOCK {block_number}/"
        f"{len(blocks)}"
    )

    print(
        f"{start_text} -> {end_text}"
    )

    print("=" * 70)


    # ========================================================
    # ONLINE SURFACE DATA
    # ========================================================

    (
        surface_norm,
        surface_raw,
        returned_dates
    ) = load_surface_block(
        start_text,
        end_text
    )


    print(
        "Online surface shape:",
        surface_norm.shape
    )


    # ========================================================
    # CREATE SAMPLES
    # ========================================================

    x_samples = []

    y_samples = []


    for day_number in range(
        len(block_dates)
    ):


        actual_date = (
            block_dates[
                day_number
            ]
        )


        print(
            f"Preparing embedding validation "
            f"{actual_date} "
            f"({day_number + 1}/"
            f"{len(block_dates)})"
        )


        # ====================================================
        # GLORYS TARGET
        # ====================================================

        target_grid = (

            validation_temperature

            .isel(
                time=
                block_start
                + day_number
            )

            .sel(
                depth=DEPTHS
            )

            .values

            .astype(
                np.float32
            )
        )


        valid_samples = 0

        attempts = 0


        # ====================================================
        # RANDOM SPATIAL SAMPLES
        # ====================================================

        while (

            valid_samples
            < SAMPLES_PER_DAY

            and

            attempts
            < SAMPLES_PER_DAY * 20

        ):

            attempts += 1


            i = random.randint(

                HALF_PATCH,

                len(TARGET_LAT)
                - HALF_PATCH
                - 1

            )


            j = random.randint(

                HALF_PATCH,

                len(TARGET_LON)
                - HALF_PATCH
                - 1

            )


            # =================================================
            # TARGET PROFILE
            # =================================================

            target_profile = target_grid[
                :,
                i,
                j
            ]


            if not np.all(
                np.isfinite(
                    target_profile
                )
            ):

                continue


            # =================================================
            # SURFACE PATCH
            # =================================================

            patch = extract_patch(

                surface_norm[
                    day_number
                ],

                i,

                j

            )


            if patch is None:

                continue


            if patch.shape != (
                7,
                9,
                9
            ):

                continue


            if not np.all(
                np.isfinite(
                    patch
                )
            ):

                continue


            # =================================================
            # SAVE SAMPLE
            # =================================================

            x_samples.append(
                patch.astype(
                    np.float32
                )
            )


            y_samples.append(
                target_profile.astype(
                    np.float32
                )
            )


            valid_samples += 1


    # ========================================================
    # BLOCK CHECK
    # ========================================================

    if len(
        x_samples
    ) == 0:

        print(
            "WARNING: No valid samples "
            "for this block."
        )


        del surface_norm

        del surface_raw

        continue


    # ========================================================
    # NUMPY ARRAYS
    # ========================================================

    x_array = np.stack(
        x_samples
    ).astype(
        np.float32
    )


    y_array = np.stack(
        y_samples
    ).astype(
        np.float32
    )


    print()

    print(
        "Block validation input:",
        x_array.shape
    )

    print(
        "Block validation target:",
        y_array.shape
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    x_tensor = torch.from_numpy(
        x_array
    )


    block_predictions = []


    model.eval()


    with torch.no_grad():

        for start in range(
            0,
            len(x_tensor),
            256
        ):


            end = min(
                start + 256,
                len(x_tensor)
            )


            batch = x_tensor[
                start:end
            ]


            output = model(
                batch.to(
                    DEVICE
                )
            )


            block_predictions.append(
                output
                .cpu()
                .numpy()
            )


    block_predictions = np.concatenate(
        block_predictions,
        axis=0
    )


    # ========================================================
    # STORE
    # ========================================================

    all_predictions.append(
        block_predictions
    )


    all_targets.append(
        y_array
    )


    # ========================================================
    # BLOCK METRICS
    # ========================================================

    block_difference = (
        block_predictions
        - y_array
    )


    block_mae = np.mean(
        np.abs(
            block_difference
        )
    )


    block_rmse = np.sqrt(
        np.mean(
            block_difference ** 2
        )
    )


    block_bias = np.mean(
        block_difference
    )


    print()

    print(
        f"Block MAE  : "
        f"{block_mae:.4f} °C"
    )


    print(
        f"Block RMSE : "
        f"{block_rmse:.4f} °C"
    )


    print(
        f"Block Bias : "
        f"{block_bias:.4f} °C"
    )


    # ========================================================
    # FREE MEMORY
    # ========================================================

    del surface_norm

    del surface_raw

    del x_samples

    del y_samples

    del x_array

    del y_array

    del x_tensor

    del block_predictions


# ============================================================
# VERIFY PREDICTIONS
# ============================================================

if len(
    all_predictions
) == 0:

    raise RuntimeError(
        "No validation predictions "
        "were produced."
    )


# ============================================================
# COMBINE BLOCKS
# ============================================================

predictions = np.concatenate(
    all_predictions,
    axis=0
)


targets = np.concatenate(
    all_targets,
    axis=0
)


print()
print("=" * 70)
print("CALCULATING FINAL EMBEDDING MODEL METRICS")
print("=" * 70)


# ============================================================
# DIFFERENCE
# ============================================================

difference = (
    predictions
    - targets
)


# ============================================================
# OVERALL METRICS
# ============================================================

overall_mae = np.mean(
    np.abs(
        difference
    )
)


overall_rmse = np.sqrt(
    np.mean(
        difference ** 2
    )
)


overall_bias = np.mean(
    difference
)


# ============================================================
# DEPTH-WISE METRICS
# ============================================================

mae_depth = np.mean(
    np.abs(
        difference
    ),
    axis=0
)


rmse_depth = np.sqrt(
    np.mean(
        difference ** 2,
        axis=0
    )
)


bias_depth = np.mean(
    difference,
    axis=0
)


# ============================================================
# SAVE RESULTS
# ============================================================

np.savez(
    RESULT_PATH,

    depths=DEPTHS,

    mae=mae_depth,

    rmse=rmse_depth,

    bias=bias_depth,

    predictions=predictions,

    targets=targets
)


# ============================================================
# FINAL REPORT
# ============================================================

print()

print("=" * 70)

print(
    "OCEANEMBED EMBEDDING MODEL "
    "2024 VALIDATION COMPLETE"
)

print("=" * 70)


print(
    "Total validation samples:",
    len(predictions)
)


print()

print(
    f"Overall MAE  : "
    f"{overall_mae:.4f} °C"
)


print(
    f"Overall RMSE : "
    f"{overall_rmse:.4f} °C"
)


print(
    f"Overall Bias : "
    f"{overall_bias:.4f} °C"
)


print()

print(
    "DEPTH-WISE RESULTS"
)


print(
    "-" * 60
)


print(
    f"{'Depth (m)':>12}"
    f"{'MAE (°C)':>15}"
    f"{'RMSE (°C)':>15}"
    f"{'Bias (°C)':>15}"
)


print(
    "-" * 60
)


for depth, mae, rmse, bias in zip(

    DEPTHS,

    mae_depth,

    rmse_depth,

    bias_depth

):

    print(

        f"{depth:12.0f}"
        f"{mae:15.4f}"
        f"{rmse:15.4f}"
        f"{bias:15.4f}"

    )


print(
    "-" * 60
)


print()

print(
    "Results saved to:"
)


print(
    RESULT_PATH
)


print(
    "=" * 70
)


# ============================================================
# CLOSE GLORYS
# ============================================================

glorys.close()