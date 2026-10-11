"""Local embedding service (loopback :8140): NVIDIA Nemotron-3-Embed-1B (OpenMDW-1.1, commercial use OK).
Mean pooling over the last hidden state, L2-normalised; 'query: ' / 'passage: ' prefixes as the model card says.
Runs from the local Hugging Face cache only (HF_HUB_OFFLINE=1)."""
import argparse
import time

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "nvidia/Nemotron-3-Embed-1B-BF16"
app = FastAPI()
S = {"model": None, "tok": None, "ready": False, "device": "cpu"}


def load():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.bfloat16 if dev == "cuda" else torch.float32, attn_implementation="sdpa").to(dev)
    model.eval()
    S.update(model=model, tok=tok, device=dev, ready=True)


@torch.inference_mode()
def encode(texts, kind):
    prefix = "query: " if kind == "query" else "passage: "
    out = []
    for i in range(0, len(texts), 8):
        enc = S["tok"]([prefix + t for t in texts[i:i + 8]], max_length=2048, truncation=True, padding=True, return_tensors="pt")
        enc = {k: v.to(S["device"]) for k, v in enc.items()}
        h = S["model"](**enc).last_hidden_state
        m = enc["attention_mask"]
        h = h.masked_fill(~m[..., None].bool(), 0.0)
        e = h.sum(1) / m.sum(1)[..., None]
        out.append(F.normalize(e.float(), p=2, dim=-1).cpu())
    return torch.cat(out).tolist()


@app.get("/health")
def health():
    return {"ok": S["ready"], "model": MODEL_ID, "device": S["device"]}


@app.post("/embed")
async def embed(req: Request):
    d = await req.json()
    t = time.perf_counter()
    vecs = encode(d["texts"], d.get("kind", "passage"))
    return {"vectors": vecs, "dim": len(vecs[0]) if vecs else 0, "ms": (time.perf_counter() - t) * 1000, "model": MODEL_ID}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    a = ap.parse_args()
    load()
    encode(["warm up"], "query")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")
