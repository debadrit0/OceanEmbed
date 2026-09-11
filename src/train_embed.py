import os
import random

from datetime import date, timedelta

import numpy as np
import torch
import torch.nn as nn
import xarray as xr

from torch.utils.data import TensorDataset, DataLoader

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

TRAIN_YEAR = 2023

BLOCK_DAYS = 30

SAMPLES_PER_DAY = 256

BATCH_SIZE = 128

# Improved training
EPOCHS_PER_BLOCK = 4

LEARNING_RATE = 0.0003

WEIGHT_DECAY = 1e-5

MIN_LEARNING_RATE = 5e-5

SEED = 42

GLORYS_PATH = os.path.join(
    "data",
    "processed",
    "glorys_full.nc"
)

CHECKPOINT_DIR = os.path.join(
    "checkpoints"
)

# IMPORTANT:
# New checkpoint so the original 49k model remains untouched.
CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "oceanembed_embedding_v2_2023.pt"
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


# ============================================================
# THERMOCLINE DEPTH WEIGHTS
# ============================================================

DEPTH_WEIGHTS = np.array(
    [
        1.00,   # 0
        1.00,   # 5
        1.00,   # 10
        1.00,   # 20
        1.00,   # 30
        1.50,   # 50
        1.75,   # 75
        2.00,   # 100
        1.75,   # 125
        1.50,   # 150
        1.00,   # 200
        1.00,   # 300
        1.00,   # 500
        1.00,   # 700
        1.00,   # 1000
    ],
    dtype=np.float32
)


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
# DIRECTORIES
# ============================================================

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD GLORYS
# ============================================================

print()
print("=" * 70)
print("OPENING GLORYS")
print("=" * 70)


if not os.path.exists(
    GLORYS_PATH
):

    raise FileNotFoundError(
        "GLORYS dataset not found:\n"
        + GLORYS_PATH
    )


glorys = xr.open_dataset(
    GLORYS_PATH
)


if "temperature" not in glorys.data_vars:

    raise RuntimeError(
        "temperature variable not found in GLORYS. "
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


# ============================================================
# VERIFY DEPTHS
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
            "not found in GLORYS."
        )


# ============================================================
# TRAINING PERIOD
# ============================================================

train_start = np.datetime64(
    f"{TRAIN_YEAR}-01-01"
)

train_end = np.datetime64(
    f"{TRAIN_YEAR}-12-31"
)


train_temperature = temperature.sel(
    time=slice(
        str(train_start),
        str(train_end)
    )
)


training_dates = [

    date(
        TRAIN_YEAR,
        1,
        1
    )
    + timedelta(
        days=i
    )

    for i in range(
        len(
            train_temperature.time
        )
    )

]


print()

print(
    "Training period:",
    str(
        train_temperature.time.values[0]
    ),
    "->",
    str(
        train_temperature.time.values[-1]
    )
)

print(
    "Training days:",
    len(training_dates)
)


# ============================================================
# CREATE TRAINING BLOCKS
# ============================================================

blocks = []

start_index = 0


while start_index < len(
    training_dates
):

    end_index = min(
        start_index + BLOCK_DAYS,
        len(training_dates)
    )

    blocks.append(
        (
            start_index,
            end_index
        )
    )

    start_index = end_index


print(
    "Training blocks:",
    len(blocks)
)


# ============================================================
# TARGET NORMALIZATION
# ============================================================
#
# Calculate mean/std ONLY from 2023 GLORYS training data.
#
# We process one day at a time to avoid loading the entire
# 2023 temperature field into memory at once.
# ============================================================

print()
print("=" * 70)
print("CALCULATING 2023 TARGET NORMALIZATION")
print("=" * 70)

sum_values = np.zeros(
    len(DEPTHS),
    dtype=np.float64
)

sum_squared_values = np.zeros(
    len(DEPTHS),
    dtype=np.float64
)

count_values = np.zeros(
    len(DEPTHS),
    dtype=np.int64
)


for time_index in range(
    len(training_dates)
):

    if (
        time_index % 30 == 0
    ):

        print(
            f"Normalization statistics: "
            f"{time_index + 1}/"
            f"{len(training_dates)} days"
        )


    daily_temperature = (
        train_temperature
        .isel(
            time=time_index
        )
        .sel(
            depth=DEPTHS
        )
        .values
        .astype(
            np.float32
        )
    )


    # daily_temperature shape:
    # [15, 101, 241]

    for depth_index in range(
        len(DEPTHS)
    ):

        values = daily_temperature[
            depth_index
        ]

        valid = values[
            np.isfinite(values)
        ]

        if len(valid) == 0:

            continue

        valid64 = valid.astype(
            np.float64
        )

        sum_values[
            depth_index
        ] += np.sum(
            valid64
        )

        sum_squared_values[
            depth_index
        ] += np.sum(
            valid64 ** 2
        )

        count_values[
            depth_index
        ] += len(valid)


target_mean = (
    sum_values
    / np.maximum(
        count_values,
        1
    )
)

target_variance = (
    (
        sum_squared_values
        / np.maximum(
            count_values,
            1
        )
    )
    - target_mean ** 2
)

target_variance = np.maximum(
    target_variance,
    1e-8
)

target_std = np.sqrt(
    target_variance
)

target_std = np.maximum(
    target_std,
    1e-4
)


print()
print(
    "2023 target normalization statistics:"
)

print(
    "-" * 70
)

print(
    f"{'Depth (m)':>12}"
    f"{'Mean (°C)':>18}"
    f"{'Std (°C)':>18}"
)

print(
    "-" * 70
)

for depth, mean_value, std_value in zip(
    DEPTHS,
    target_mean,
    target_std
):

    print(
        f"{depth:12.0f}"
        f"{mean_value:18.6f}"
        f"{std_value:18.6f}"
    )

print(
    "-" * 70
)


# ============================================================
# MODEL
# ============================================================

model = (
    OceanEmbedEmbeddingModel()
    .to(DEVICE)
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    "Embedding dimension:",
    model.EMBEDDING_DIM
)

print(
    "Model parameters:",
    parameter_count
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# DEPTH WEIGHTS TENSOR
# ============================================================

depth_weights_tensor = torch.tensor(
    DEPTH_WEIGHTS,
    dtype=torch.float32,
    device=DEVICE
)


# ============================================================
# LEARNING-RATE SCHEDULER
# ============================================================

total_epochs = (
    len(blocks)
    * EPOCHS_PER_BLOCK
)


scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max(
        1,
        total_epochs
    ),
    eta_min=MIN_LEARNING_RATE
)


# ============================================================
# RESUME SUPPORT
# ============================================================

start_block = 0


if os.path.exists(
    CHECKPOINT_PATH
):

    print()
    print(
        "V2 checkpoint found:"
    )

    print(
        CHECKPOINT_PATH
    )


    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # Verify normalization compatibility
    # --------------------------------------------------------

    checkpoint_mean = np.asarray(
        checkpoint.get(
            "target_mean",
            []
        ),
        dtype=np.float32
    )

    checkpoint_std = np.asarray(
        checkpoint.get(
            "target_std",
            []
        ),
        dtype=np.float32
    )


    if (
        checkpoint_mean.shape
        != target_mean.astype(
            np.float32
        ).shape
        or
        checkpoint_std.shape
        != target_std.astype(
            np.float32
        ).shape
    ):

        raise RuntimeError(
            "Existing V2 checkpoint has incompatible "
            "target normalization statistics."
        )


    if not np.allclose(
        checkpoint_mean,
        target_mean.astype(
            np.float32
        ),
        atol=1e-5
    ):

        raise RuntimeError(
            "Existing V2 checkpoint target_mean "
            "does not match current 2023 statistics."
        )


    if not np.allclose(
        checkpoint_std,
        target_std.astype(
            np.float32
        ),
        atol=1e-5
    ):

        raise RuntimeError(
            "Existing V2 checkpoint target_std "
            "does not match current 2023 statistics."
        )


    model.load_state_dict(
        checkpoint[
            "model_state"
        ]
    )


    if (
        "optimizer_state"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state"
            ]
        )


    if (
        "scheduler_state"
        in checkpoint
    ):

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state"
            ]
        )


    start_block = checkpoint.get(
        "next_block",
        0
    )


    print(
        "Resuming from block:",
        start_block + 1
    )


# ============================================================
# TRAINING SUMMARY
# ============================================================

print()
print("=" * 70)
print("OCEANEMBED EMBEDDING MODEL V2 TRAINING")
print("=" * 70)

print(
    "Training year:",
    TRAIN_YEAR
)

print(
    "Validation year:",
    "2024"
)

print(
    "Training days:",
    len(training_dates)
)

print(
    "Blocks:",
    len(blocks)
)

print(
    "Samples/day:",
    SAMPLES_PER_DAY
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs/block:",
    EPOCHS_PER_BLOCK
)

print(
    "Initial learning rate:",
    LEARNING_RATE
)

print(
    "Minimum learning rate:",
    MIN_LEARNING_RATE
)

print(
    "Embedding dimension:",
    model.EMBEDDING_DIM
)

print(
    "Parameter count:",
    parameter_count
)

print(
    "Thermocline weighting:",
    "50-150 m emphasized"
)

print(
    "Target normalization:",
    "2023-only per-depth"
)

print(
    "Checkpoint:",
    CHECKPOINT_PATH
)

print("=" * 70)


# ============================================================
# TRAINING LOOP
# ============================================================

for block_number, (
    block_start,
    block_end
) in enumerate(
    blocks[start_block:],
    start=start_block + 1
):


    block_dates = training_dates[
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
    print(
        "#" * 70
    )

    print(
        f"V2 BLOCK {block_number}/"
        f"{len(blocks)}"
    )

    print(
        f"{start_text} -> {end_text}"
    )

    print(
        "#" * 70
    )


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
        "Surface data:",
        surface_norm.shape
    )


    # ========================================================
    # TRAINING SAMPLES
    # ========================================================

    x_samples = []

    y_samples = []


    for day_number in range(
        len(block_dates)
    ):


        actual_date = block_dates[
            day_number
        ]


        print(
            f"Preparing "
            f"{actual_date} "
            f"({day_number + 1}/"
            f"{len(block_dates)})"
        )


        # ====================================================
        # GLORYS TARGET
        # ====================================================

        target_grid = (
            train_temperature
            .isel(
                time=block_start + day_number
            )
            .sel(
                depth=DEPTHS
            )
            .values
            .astype(
                np.float32
            )
        )


        # ====================================================
        # RANDOM SAMPLES
        # ====================================================

        valid_samples = 0

        attempts = 0


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

            target_profile = (
                target_grid[
                    :,
                    i,
                    j
                ]
            )


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
            # STORE SAMPLE
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
    # SAMPLE CHECK
    # ========================================================

    if len(
        x_samples
    ) == 0:

        print(
            "WARNING: No valid samples "
            "for this block."
        )

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
        "Training input:",
        x_array.shape
    )

    print(
        "Training target:",
        y_array.shape
    )


    # ========================================================
    # TARGET NORMALIZATION
    # ========================================================

    y_normalized = (
        (
            y_array
            - target_mean.astype(
                np.float32
            )
        )
        /
        target_std.astype(
            np.float32
        )
    )


    print(
        "Normalized target range:",
        float(
            np.min(
                y_normalized
            )
        ),
        "to",
        float(
            np.max(
                y_normalized
            )
        )
    )


    # ========================================================
    # PYTORCH DATASET
    # ========================================================

    x_tensor = torch.from_numpy(
        x_array
    )


    y_tensor = torch.from_numpy(
        y_normalized.astype(
            np.float32
        )
    )


    dataset = TensorDataset(
        x_tensor,
        y_tensor
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()


    block_loss = None


    for epoch in range(
        EPOCHS_PER_BLOCK
    ):


        running_loss = 0.0

        batch_count = 0


        current_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]


        print()
        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{EPOCHS_PER_BLOCK}"
            f"  LR={current_lr:.7f}"
        )


        for batch_x, batch_y in loader:


            batch_x = batch_x.to(
                DEVICE
            )


            batch_y = batch_y.to(
                DEVICE
            )


            optimizer.zero_grad()


            predictions = model(
                batch_x
            )


            # ------------------------------------------------
            # Smooth L1 loss per depth
            # ------------------------------------------------

            element_loss = nn.functional.smooth_l1_loss(
                predictions,
                batch_y,
                reduction="none"
            )


            # ------------------------------------------------
            # Thermocline-weighted loss
            # ------------------------------------------------

            weighted_loss = (
                element_loss
                * depth_weights_tensor
            )


            loss = (
                weighted_loss.sum()
                /
                (
                    batch_y.shape[0]
                    *
                    depth_weights_tensor.sum()
                )
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
            )


            batch_count += 1


        epoch_loss = (
            running_loss
            /
            max(
                1,
                batch_count
            )
        )


        block_loss = epoch_loss


        print(
            f"Epoch loss = "
            f"{epoch_loss:.6f}"
        )


        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()


        new_lr = optimizer.param_groups[
            0
        ][
            "lr"
        ]


        print(
            f"Next learning rate = "
            f"{new_lr:.7f}"
        )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "scheduler_state":
            scheduler.state_dict(),

        "next_block":
            block_number,

        "year":
            TRAIN_YEAR,

        "depths":
            DEPTHS.tolist(),

        "depth_weights":
            DEPTH_WEIGHTS.tolist(),

        "target_mean":
            target_mean.astype(
                np.float32
            ),

        "target_std":
            target_std.astype(
                np.float32
            ),

        "embedding_dim":
            model.EMBEDDING_DIM,

        "parameter_count":
            parameter_count,

        "learning_rate":
            LEARNING_RATE,

        "minimum_learning_rate":
            MIN_LEARNING_RATE,

        "epochs_per_block":
            EPOCHS_PER_BLOCK,

        "last_loss":
            float(
                block_loss
            ),
    }


    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )


    print()
    print(
        "V2 checkpoint saved:"
    )

    print(
        CHECKPOINT_PATH
    )


    # ========================================================
    # RELEASE MEMORY
    # ========================================================

    del surface_norm
    del surface_raw
    del x_samples
    del y_samples
    del x_array
    del y_array
    del y_normalized
    del x_tensor
    del y_tensor
    del dataset
    del loader


    print(
        f"V2 block {block_number} complete."
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("2023 OCEANEMBED EMBEDDING V2 TRAINING COMPLETE")
print("=" * 70)

print(
    "Model saved to:"
)

print(
    CHECKPOINT_PATH
)

print(
    "Training year:",
    TRAIN_YEAR
)

print(
    "Validation year remains:",
    "2024"
)

print("=" * 70)


glorys.close()