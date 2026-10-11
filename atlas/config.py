"""Single source of product configuration."""
import os
from pathlib import Path

ASSISTANT_NAME = os.environ.get("ATLAS_NAME", "Atlas")
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "atlas.db"
OUT_DIR = ROOT / "out"

# "Today" for the demo world. Fixed so every number is reproducible.
AS_OF = os.environ.get("ATLAS_AS_OF", "2026-10-08")

LLM_URL = os.environ.get("ATLAS_LLM_URL", "http://127.0.0.1:8101")
SPEECH_URL = os.environ.get("ATLAS_SPEECH_URL", "http://127.0.0.1:8110")
KEV_URL = os.environ.get("ATLAS_KEV_URL", "http://127.0.0.1:8120")

# Optional URL encoded as a QR code on the Executive Brief. Empty = no QR.
BRIEF_QR_URL = os.environ.get("ATLAS_QR_URL", "")

PERSONAS = {
    "enterprise": {
        "org": "Meridian Dynamics",
        "role": "CRO / CFO",
        "voice": "Sofia",
    },
    "university": {
        "org": "Alta Vista University",
        "role": "President / VP Research",
        "voice": "Aria",
    },
    "developer": {
        "org": "Meridian Dynamics",
        "role": "engineering team",
        "voice": "Sofia",
    },
}

MC_SIMS = 10_000
SEED = 42

# v2 features. Each can be switched off with an environment variable (e.g. ATLAS_ORB=0) if it misbehaves;
# the v1 demo keeps working with all of them off.
def _flag(name: str, default: bool = True) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() not in ("0", "false", "off", "no", "")


FEATURES = {
    "orb": _flag("ATLAS_ORB"),              # voice orb (WebGL / Canvas 2D)
    "brain": _flag("ATLAS_BRAIN"),          # second brain view + knowledge graph provenance
    "docs": _flag("ATLAS_DOCS"),            # document index + search_knowledge tool
    "feed": _flag("ATLAS_FEED"),            # "Feed the brain" (LLM writes a new synthetic document)
    "developer": _flag("ATLAS_DEV"),        # Developer persona (code generation + sandbox)
    "finetune": _flag("ATLAS_FINETUNE"),    # use a fine-tuned decision model if one was kept
}
