"""Paths, model ids, and thresholds -- the single source of truth.

Nothing else in the package hard-codes a model name or a magic number. Thresholds
that Phase 2 calibrates from labelled data are declared here as `None` so that an
uncalibrated run fails loudly instead of silently using a guess.
"""

from __future__ import annotations

import os
from pathlib import Path

# huggingface_hub >=1.0 routes downloads through Xet by default. On this machine
# that transport hangs at 0 B/s with the process alive and no error; forcing the
# classic CDN path restores normal throughput. Set before any transformers or
# huggingface_hub import, which is why it lives at the top of the config module.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# --- paths -------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

FRAMES = DATA / "frames"
INDEX = DATA / "index"
GALLERY = DATA / "gallery"
EVAL = DATA / "eval"
SPIKE = DATA / "spike"  # Phase 0 fixtures only

DB_PATH = INDEX / "sceneseek.db"
IMAGE_VECS = INDEX / "image_vecs.npy"
FACE_VECS = INDEX / "face_vecs.npy"


def frames_dir(film: str) -> Path:
    return FRAMES / film


def manual_dir(film: str) -> Path:
    """Drop-in folder for hand-filled coverage gaps. Honoured by every phase."""
    return FRAMES / film / "manual"


# --- face detection + recognition --------------------------------------------

# "insightface" (primary), "onnx" (fallback A: same buffalo_l weights, no package),
# "facenet" (fallback B: pure torch). Phase 2's FaceBackend protocol makes this a
# one-line swap.
FACE_BACKEND = "insightface"
FACE_MODEL = "buffalo_l"  # RetinaFace det_10g + ArcFace w600k_r50, ~330 MB
FACE_MODEL_ROOT = Path("~/.insightface").expanduser()
FACE_MODULES = ("detection", "recognition")  # skip genderage/landmark, unused here
FACE_DET_SIZE = (640, 640)
FACE_DET_THRESH = 0.5

# onnxruntime execution providers, in priority order. Plain CPU is already fast
# enough at ~450 images; CoreML is listed first only where it is actually present.
ONNX_PROVIDERS = ("CPUExecutionProvider",)

# --- visual encoder ----------------------------------------------------------

VISUAL_MODEL = "google/siglip2-base-patch16-384"
# so400m is the quality upgrade path, not the default: its fp32 checkpoint is
# 4.54 GB against base's 1.5 GB, and Phase 0 measured ~0.9-3.7 MB/s to the HF CDN
# on this machine. Phase 5 should re-run the eval against it before concluding the
# ranker is the bottleneck on description-critical queries.
VISUAL_MODEL_FALLBACKS = (
    "google/siglip2-so400m-patch14-384",
    "google/siglip-so400m-patch14-384",
    "google/siglip-base-patch16-224",
)
# fp32 on MPS: ~3.5 GB is comfortable in 24 GB, throughput is irrelevant at this
# corpus size, and fp32 sidesteps the fp16-on-MPS dtype bugs in vision towers.
VISUAL_DTYPE = "float32"
VISUAL_BATCH = 8
TEXT_MAX_LENGTH = 64  # SigLIP tokenizers want fixed-width padding

DEVICE = "mps"

# --- thresholds --------------------------------------------------------------

# Calibrated in Phase 2 from the genuine/impostor distributions, not guessed.
# `None` until `data/gallery/<film>/calibration.yaml` exists.
FACE_SIM_THRESHOLD: float | None = None
TARGET_IDENTITY_PRECISION = 0.98

# Agglomerative clustering over L2-normalized face embeddings (cosine, average
# linkage). Tuned on one film in Phase 2.
CLUSTER_DISTANCE_THRESHOLD = 0.55
CLUSTER_MIN_SIZE = 3

# Perceptual-hash dedupe (Phase 1).
PHASH_HAMMING_MAX = 6

# --- acquisition -------------------------------------------------------------

TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/original"
SUPPLEMENT_RATE_LIMIT_S = 1.0
USER_AGENT = "SceneSeek/0.1 (research prototype; https://github.com/sceneseek) python-urllib"
MIN_FRAMES_PER_FILM = 60
