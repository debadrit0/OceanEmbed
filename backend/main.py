from pathlib import Path
import sys
import numpy as np
import torch
import xarray as xr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.model import OceanEmbed
from src.config import DEPTHS_M, LATS, LONS

app=FastAPI(title="OceanEmbed AI", version="2.0")
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
MODEL=OceanEmbed().to(DEVICE)
CKPT=Path(__file__).resolve().parents[1]/"artifacts"/"oceanembed.pt"
MODEL_READY=CKPT.exists()
if MODEL_READY:
    MODEL.load_state_dict(torch.load(CKPT,map_location=DEVICE)["model"])
MODEL.eval()

class PredictRequest(BaseModel):
    features: list[list[list[float]]]

@app.get("/health")
def health():
    return {"status":"ok","model_ready":MODEL_READY,"device":DEVICE,
            "domain":{"lat_min":float(LATS.min()),"lat_max":float(LATS.max()),"lon_min":float(LONS.min()),"lon_max":float(LONS.max())},
            "depths_m":DEPTHS_M.astype(int).tolist()}

@app.post("/predict")
def predict(req: PredictRequest):
    if not MODEL_READY: raise HTTPException(503,"Train the model and put artifacts/oceanembed.pt in place.")
    a=np.asarray(req.features,dtype=np.float32)
    if a.ndim!=3 or a.shape[0]!=7: raise HTTPException(400,"features must have shape [7,H,W]")
    with torch.no_grad(): y,z=MODEL(torch.from_numpy(np.nan_to_num(a,nan=0.0))[None].to(DEVICE))
    return {"depths_m":DEPTHS_M.astype(int).tolist(),"temperature":y[0].cpu().numpy().tolist(),"embedding_shape":list(z.shape)}
