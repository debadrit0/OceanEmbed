import os
import random
from datetime import date, timedelta

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch.utils.data import TensorDataset, DataLoader

from model_fast import OceanEmbedFast
from online_year_data import (
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

EPOCHS_PER_BLOCK = 2

LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-5

SEED = 42

GLORYS_PATH = os.path.join(
    "data",
    "processed",
    "glorys_full.nc"
)

CHECKPOINT_DIR = "checkpoints"

CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
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


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# CPU
# ============================================================

DEVICE = torch.device("cpu")

torch.set_num_threads(
    max(1, min(8, os.cpu_count() or 1))
)

print("Device:", DEVICE)
print("CPU threads:", torch.get_num_threads())


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

glorys = xr.open_dataset(
    GLORYS_PATH
)

# IMPORTANT:
# Your GLORYS file calls the temperature variable
# "temperature", not "thetao".

if "temperature" not in glorys.data_vars:

    raise RuntimeError(
        "temperature not found. "
        f"Variables: {list(glorys.data_vars)}"
    )

temperature = glorys["temperature"]

print(
    "GLORYS dimensions:",
    temperature.dims
)

print(
    "GLORYS shape:",
    temperature.shape
)

print(
    "Depths in file:"
)

print(
    temperature.depth.values
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
            f"Required depth {depth} m not found."
        )


# ============================================================
# TRAINING YEAR
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

print()

print(
    "Training period:",
    str(train_temperature.time.values[0]),
    "->",
    str(train_temperature.time.values[-1])
)

training_dates = [

    date(TRAIN_YEAR, 1, 1)
    + timedelta(days=i)

    for i in range(
        len(train_temperature.time)
    )
]

print(
    "Training days:",
    len(training_dates)
)


# ============================================================
# MODEL
# ============================================================

model = OceanEmbedFast().to(
    DEVICE
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print()
print(
    "Model parameters:",
    parameter_count
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

loss_function = nn.SmoothL1Loss()


# ============================================================
# RESUME SUPPORT
# ============================================================

start_block = 0

if os.path.exists(
    CHECKPOINT_PATH
):

    print()
    print(
        "Checkpoint found:"
    )

    print(
        CHECKPOINT_PATH
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
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
# YEAR BLOCKS
# ============================================================

blocks = []

start_index = 0

while start_index < len(training_dates):

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


print()
print("=" * 70)
print("OCEANEMBED 2023 TRAINING")
print("=" * 70)

print(
    "Year:",
    TRAIN_YEAR
)

print(
    "Days:",
    len(training_dates)
)

print(
    "Blocks:",
    len(blocks)
)

print(
    "Block size:",
    BLOCK_DAYS
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
    print("#" * 70)

    print(
        f"BLOCK {block_number}/{len(blocks)}"
    )

    print(
        f"{start_text} -> {end_text}"
    )

    print("#" * 70)


    # ========================================================
    # ONLINE SURFACE DATA
    # ========================================================

    surface_norm, surface_raw, returned_dates = (
        load_surface_block(
            start_text,
            end_text
        )
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


        # ----------------------------------------------------
        # GLORYS TARGET FOR THIS DAY
        # ----------------------------------------------------

        target_grid = (
            train_temperature
            .isel(
                time=block_start + day_number
            )
            .sel(
                depth=DEPTHS
            )
            .values
            .astype(np.float32)
        )

        # shape:
        # [15, 101, 241]


        # ----------------------------------------------------
        # RANDOM SPATIAL SAMPLES
        # ----------------------------------------------------

        valid_samples = 0

        attempts = 0

        while (
            valid_samples < SAMPLES_PER_DAY
            and attempts < SAMPLES_PER_DAY * 20
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


            # ------------------------------------------------
            # TARGET PROFILE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # SURFACE PATCH
            # ------------------------------------------------

            patch = extract_patch(
                surface_norm[
                    day_number
                ],
                i,
                j
            )


            if patch is None:
                continue


            if not np.all(
                np.isfinite(
                    patch
                )
            ):

                continue


            # ------------------------------------------------
            # ADD SAMPLE
            # ------------------------------------------------

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
    # CHECK SAMPLES
    # ========================================================

    if len(x_samples) == 0:

        print(
            "WARNING: No valid samples."
        )

        continue


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
    # PYTORCH DATASET
    # ========================================================

    x_tensor = torch.from_numpy(
        x_array
    )

    y_tensor = torch.from_numpy(
        y_array
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
    # TRAIN MODEL
    # ========================================================

    model.train()


    for epoch in range(
        EPOCHS_PER_BLOCK
    ):

        running_loss = 0.0

        batch_count = 0


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


            loss = loss_function(
                predictions,
                batch_y
            )


            loss.backward()


            optimizer.step()


            running_loss += (
                loss.item()
            )

            batch_count += 1


        epoch_loss = (
            running_loss
            / batch_count
        )


        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{EPOCHS_PER_BLOCK} "
            f"Loss = "
            f"{epoch_loss:.6f}"
        )


    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    checkpoint = {

        "model_state":
            model.state_dict(),

        "optimizer_state":
            optimizer.state_dict(),

        "next_block":
            block_number,

        "year":
            TRAIN_YEAR,

        "depths":
            DEPTHS.tolist(),

        "last_loss":
            float(epoch_loss)
    }


    torch.save(
        checkpoint,
        CHECKPOINT_PATH
    )


    print()

    print(
        "Checkpoint saved:"
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
    del x_tensor
    del y_tensor
    del dataset
    del loader


    print(
        f"Block {block_number} complete."
    )


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("2023 TRAINING COMPLETE")
print("=" * 70)

print(
    "Model saved to:"
)

print(
    CHECKPOINT_PATH
)

glorys.close()