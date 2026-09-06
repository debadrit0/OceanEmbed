# OceanEmbed

OceanEmbed is a machine-learning system for reconstructing subsurface ocean temperature profiles across the North Indian Ocean using online surface ocean observations.

## Domain

- Latitude: 5°N to 30°N
- Longitude: 45°E to 105°E
- Grid: 0.25° × 0.25°
- Temporal resolution: Daily

## Surface Inputs

OceanEmbed uses seven surface variables:

1. Sea Surface Temperature (SST)
2. Sea Surface Salinity (SSS)
3. Sea Surface Height / Sea Level Anomaly (SSH/SLA)
4. Surface zonal current (U)
5. Surface meridional current (V)
6. Surface zonal wind (U)
7. Surface meridional wind (V)

## Predicted Depths

The model predicts temperature at:

0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

## Model

OceanEmbed uses a lightweight convolutional neural network operating on 7-channel 9×9 surface patches.

Training:
- 2023

Validation:
- 2024

## Validation Results

Overall 2024 validation:

- MAE: 0.8156 °C
- RMSE: 1.2104 °C
- Bias: +0.1557 °C
- Validation samples: 46,848

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt