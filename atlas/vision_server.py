"""Person detection service (RF-DETR Nano, Apache 2.0) on :8130. POST a JPEG to /detect."""
from __future__ import annotations

import argparse
import io
import time

import uvicorn
from fastapi import FastAPI, Request
from PIL import Image

app = FastAPI()
MODEL = {"m": None, "person_ids": {1}}


def load():
    from rfdetr import RFDETRNano
    from pathlib import Path
    local = Path(__file__).resolve().parent.parent / "vendor" / "vision" / "rf-detr-nano.pth"
    m = RFDETRNano(pretrain_weights=str(local)) if local.exists() else RFDETRNano()
    MODEL["m"] = m
    MODEL["person_ids"] = {1}  # predictions use COCO category ids; 1 = person
    m.predict(Image.new("RGB", (640, 360)), threshold=0.5)  # warm up
    print("RF-DETR Nano ready; person class ids", MODEL["person_ids"])


@app.get("/health")
def health():
    return {"ok": MODEL["m"] is not None}


@app.post("/detect")
async def detect(req: Request):
    img = Image.open(io.BytesIO(await req.body())).convert("RGB")
    t = time.perf_counter()
    det = MODEL["m"].predict(img, threshold=0.4)
    ms = (time.perf_counter() - t) * 1000
    w, h = img.size
    people = []
    for box, cls, conf in zip(det.xyxy, det.class_id, det.confidence):
        if int(cls) in MODEL["person_ids"]:
            x1, y1, x2, y2 = [float(v) for v in box]
            people.append({"box": [x1 / w, y1 / h, x2 / w, y2 / h], "score": float(conf)})
    return {"ok": True, "people": people, "ms": ms}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8130)
    a = ap.parse_args()
    load()
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
