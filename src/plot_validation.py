import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

RESULT_PATH = os.path.join(
    "results",
    "validation_2024.npz"
)

PLOT_DIR = os.path.join(
    "results",
    "plots"
)


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(
    RESULT_PATH
):

    raise FileNotFoundError(
        f"Validation file not found:\n"
        f"{RESULT_PATH}"
    )


os.makedirs(
    PLOT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

data = np.load(
    RESULT_PATH
)

depths = data["depths"]

mae = data["mae"]

rmse = data["rmse"]

bias = data["bias"]

predictions = data["predictions"]

targets = data["targets"]


# ============================================================
# OVERALL METRICS
# ============================================================

difference = (
    predictions - targets
)

overall_mae = np.mean(
    np.abs(difference)
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
# 1. MAE VS DEPTH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    mae,
    depths,
    marker="o"
)

plt.gca().invert_yaxis()

plt.xlabel(
    "MAE (°C)"
)

plt.ylabel(
    "Depth (m)"
)

plt.title(
    "OceanEmbed 2024 Validation - MAE vs Depth"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "mae_vs_depth.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 2. RMSE VS DEPTH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    rmse,
    depths,
    marker="o"
)

plt.gca().invert_yaxis()

plt.xlabel(
    "RMSE (°C)"
)

plt.ylabel(
    "Depth (m)"
)

plt.title(
    "OceanEmbed 2024 Validation - RMSE vs Depth"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "rmse_vs_depth.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 3. BIAS VS DEPTH
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    bias,
    depths,
    marker="o"
)

plt.axvline(
    0,
    linestyle="--"
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Bias (°C)"
)

plt.ylabel(
    "Depth (m)"
)

plt.title(
    "OceanEmbed 2024 Validation - Bias vs Depth"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "bias_vs_depth.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 4. PREDICTED VS ACTUAL
# ============================================================

flat_true = targets.flatten()

flat_pred = predictions.flatten()

# Limit plotted points so the figure remains readable.

max_points = 10000

if len(flat_true) > max_points:

    rng = np.random.default_rng(
        42
    )

    indices = rng.choice(
        len(flat_true),
        size=max_points,
        replace=False
    )

    flat_true_plot = flat_true[
        indices
    ]

    flat_pred_plot = flat_pred[
        indices
    ]

else:

    flat_true_plot = flat_true

    flat_pred_plot = flat_pred


minimum = min(
    flat_true_plot.min(),
    flat_pred_plot.min()
)

maximum = max(
    flat_true_plot.max(),
    flat_pred_plot.max()
)


plt.figure(
    figsize=(8, 8)
)

plt.scatter(
    flat_true_plot,
    flat_pred_plot,
    alpha=0.25,
    s=8
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "GLORYS Temperature (°C)"
)

plt.ylabel(
    "OceanEmbed Prediction (°C)"
)

plt.title(
    "OceanEmbed vs GLORYS - 2024 Validation"
)

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "predicted_vs_actual.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# 5. EXAMPLE TEMPERATURE PROFILE
# ============================================================

example_index = 0

example_true = targets[
    example_index
]

example_pred = predictions[
    example_index
]


plt.figure(
    figsize=(8, 7)
)

plt.plot(
    example_true,
    depths,
    marker="o",
    label="GLORYS"
)

plt.plot(
    example_pred,
    depths,
    marker="s",
    label="OceanEmbed"
)

plt.gca().invert_yaxis()

plt.xlabel(
    "Temperature (°C)"
)

plt.ylabel(
    "Depth (m)"
)

plt.title(
    "Example Ocean Temperature Profile"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "example_temperature_profile.png"
    ),
    dpi=200
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 70)

print(
    "OCEANEMBED VALIDATION PLOTS CREATED"
)

print("=" * 70)

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
    "Plot directory:"
)

print(
    PLOT_DIR
)

print()

print(
    "Created:"
)

print(
    "  mae_vs_depth.png"
)

print(
    "  rmse_vs_depth.png"
)

print(
    "  bias_vs_depth.png"
)

print(
    "  predicted_vs_actual.png"
)

print(
    "  example_temperature_profile.png"
)

print(
    "=" * 70
)