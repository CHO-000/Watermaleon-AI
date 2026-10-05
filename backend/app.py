from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from model import CLASSES, THAI_LABELS, build_model, image_to_tensor

MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.pt")))
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file missing: {MODEL_PATH}. Run train.py first.")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != CLASSES:
        raise RuntimeError("Model classes do not match the API code")
    model = build_model(architecture=checkpoint.get("architecture", "mobilenet_v3_small")).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    app.state.model = model
    app.state.device = device
    yield


app = FastAPI(title="Watermelon Leaf Disease API", version="1.0.0", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "device": str(app.state.device), "classes": CLASSES}


@app.get("/classes")
def classes():
    return [{"id": name, "name_th": THAI_LABELS[name]} for name in CLASSES]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            tensor = image_to_tensor(image).to(app.state.device)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=422, detail="Invalid image file")
    with torch.inference_mode():
        scores = torch.softmax(app.state.model(tensor), dim=1)[0].cpu().tolist()
    winner = max(range(len(scores)), key=scores.__getitem__)
    label = CLASSES[winner]
    return {
        "class_id": label,
        "class_name_th": THAI_LABELS[label],
        "confidence": round(scores[winner], 6),
        "scores": {name: round(score, 6) for name, score in zip(CLASSES, scores)},
        "note": "ผลคัดกรองจากภาพใบแตงโม ไม่ใช่การวินิจฉัยยืนยัน",
    }
