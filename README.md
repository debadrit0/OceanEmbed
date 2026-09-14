# OceanEmbed: High-Resolution Subsurface Ocean Temperature Reconstruction

OceanEmbed is a physics-informed deep learning operational platform for reconstructing 3D subsurface ocean temperature profiles (0 to 1000 meters across 15 standard oceanographic depths) using multi-satellite surface remote sensing observations across the North Indian Ocean domain (5°N–30°N, 45°E–105°E).

---

## 🌊 Overview & Scientific Motivation

Satellite sensors accurately capture sea surface dynamics (thermal infrared SST, radiometer salinity, radar altimetry SSH, scatterometer winds, and total surface currents). However, deep subsurface ocean thermal structures (thermocline depth, internal waves, heat content, acoustic ducts) historically require expensive in-situ instruments such as Argo floats, CTD casts, and moored buoys.

**OceanEmbed** bridges this gap through a dual deep-learning ensemble that maps high-dimensional 7-channel spatial patches into compact latent oceanographic embeddings, reconstructing vertical temperature profiles down to 1000m with sub-degree accuracy.

---

## 🚀 Key Features & Architectural Highlights

- **Dual Ensemble Architecture**:
  - **OceanEmbed V2 Embedding (64D Latent Space)**: Deep spatial convolutional encoder with adaptive average pooling and target standardization against ocean climatology.
  - **OceanEmbed Fast (32D Latent Baseline)**: Rapid-inference lightweight CNN optimized for real-time operational mapping.
  - **Physics-Informed Ensemble Blend**: 70% V2 Embedding + 30% Fast Baseline achieving **MAE 0.7222 °C** and Pearson Correlation **R = 0.9905** across independent validation observations.
- **7 Satellite Surface Input Channels (9×9 Spatial Patches at 0.25° Resolution)**:
  1. Sea Surface Temperature (SST) — OSTIA Level-4 Operational Foundation SST (°C)
  2. Sea Surface Salinity (SSS) — SMOS/SMAP Multi-Mission Surface Salinity (PSU)
  3. Sea Surface Height / Sea Level Anomaly (SSH / SLA) — DUACS Multi-Satellite Altimetry (m)
  4. Zonal Surface Current ($u$) — Total Geostrophic + Ekman Current (m/s)
  5. Meridional Surface Current ($v$) — Total Geostrophic + Ekman Current (m/s)
  6. Zonal 10m Wind ($u$) — ASCAT MetOp-B Scatterometer (m/s)
  7. Meridional 10m Wind ($v$) — ASCAT MetOp-B Scatterometer (m/s)
- **15 Standard Target Depths**:
  `[0m, 5m, 10m, 20m, 30m, 50m, 75m, 100m, 125m, 150m, 200m, 300m, 500m, 700m, 1000m]`
- **Interactive Web Platform**:
  - **Map Predictor**: Regional temperature contour maps, live point inspection, and depth slicing.
  - **Manual Simulation**: Parameter-driven sensitivity analysis with 1D vertical temperature splines and dynamic 2D zonal thermal cross-sections.
  - **Live Math & Latent Diagnostics**: Real-time layer-by-layer tensor activations, cosine similarities, energy conservation metrics, and 64-dimensional latent embedding heatmaps.

---

## ⚡ Quick Start (Run Immediately)

The repository includes pre-trained lightweight model checkpoints (~800 KB total in `checkpoints/`), allowing you to launch and run the interactive platform immediately without downloading multi-gigabyte external datasets.

### 1. Clone the repository
```bash
git clone https://github.com/debadrit0/OceanEmbed.git
cd OceanEmbed
```

### 2. Create and activate a virtual environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the application
```bash
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

---

## 📊 Dataset Acquisition & Local Dataset Creation

You can construct the full dataset locally on your machine either using live Copernicus Marine Service satellite feeds or via synthetic physics-consistent simulation.

### Option A: Synthetic Generation (Instant, No Credentials Required)
If you want to immediately generate training/validation datasets and test the training pipeline without needing a Copernicus Marine account:
```bash
# Generate 3,000 multi-channel 9x9 patches with 15 depth targets:
python create_dataset.py --synthetic --num-samples 3000
```
This produces:
- `data/processed/train_patches.npz` (2,400 training samples)
- `data/processed/val_patches.npz` (600 validation samples)

### Option B: Real Copernicus Marine Satellite Downloads
To download operational satellite observations from Copernicus Marine Service (CMEMS):

1. Register for a free account at [marine.copernicus.eu](https://marine.copernicus.eu).
2. Authenticate locally:
   ```bash
   copernicusmarine login
   ```
3. Download the raw datasets:
   ```bash
   # Download a 3-day sample:
   python download_data.py --sample

   # Or download a custom date range:
   python download_data.py --start-date 2023-01-01 --end-date 2023-01-31
   ```
4. Process raw NetCDF files into spatial patches:
   ```bash
   python create_dataset.py
   ```

---

## 🏋️ Model Training

Train the dual architectures from scratch using your processed datasets:

```bash
# Train both models (Fast baseline + V2 Embedding):
python train.py --epochs 15 --batch-size 64

# Train only the Fast model:
python train.py --model fast --epochs 10

# Train with GPU acceleration:
python train.py --device cuda --epochs 20
```

Trained checkpoints are automatically saved to `checkpoints/`:
- `checkpoints/oceanembed_fast_2023.pt`
- `checkpoints/oceanembed_embedding_v2_2023.pt`

---

## 📈 Performance & Benchmark Metrics

Evaluated across independent held-out operational test profiles:

| Model Architecture | Latent Dim | Overall MAE (°C) | Thermocline MAE (100m) | Correlation ($R$) | Parameter Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OceanEmbed Fast** | 32D | 0.8156 | 1.1240 | 0.9842 | ~24,000 |
| **OceanEmbed V2 Embedding** | 64D | 0.7502 | 0.9850 | 0.9881 | ~52,000 |
| **Dual Ensemble (V2 + Fast)** | **Combined** | **0.7222** | **0.8910** | **0.9905** | **~76,000** |

---

## 📂 Project Structure

```
OceanEmbed/
├── app.py                   # Main Flask application & inference server
├── download_data.py         # CMEMS data acquisition tool (CLI)
├── create_dataset.py        # Patch generation & dataset processor
├── train.py                 # End-to-end deep learning training pipeline
├── requirements.txt         # Core dependencies
├── .gitignore               # Excludes large binaries/data, tracks checkpoints
├── checkpoints/             # Trained model weights (~800 KB total)
│   ├── oceanembed_fast_2023.pt
│   └── oceanembed_embedding_v2_2023.pt
├── src/                     # Core neural network architectures & utilities
│   ├── config.py            # Domain grid boundaries & depth levels
│   ├── model_fast.py        # 32D Latent Baseline CNN architecture
│   ├── model_embed.py       # 64D Latent V2 Spatial CNN architecture
│   ├── ocean_mask.py        # Land-sea masking & geographic filters
│   └── online_year_data.py  # Copernicus online observation pipelines
├── templates/               # Web application UI
│   └── index.html           # Full interactive platform dashboard
└── scripts/                 # Auxiliary download & environment utilities
    ├── download_guide.sh
    └── download_oscar.sh
```

---

## 📜 License

This project is licensed under the Apache 2.0 License.