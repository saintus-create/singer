"""Download pretrained models from Hugging Face Hub if not already present."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRETRAINED_DIR = ROOT / "pretrained_models"
MODEL_DIR_SVS = PRETRAINED_DIR / "SoulX-Singer"
MODEL_DIR_PREPROCESS = PRETRAINED_DIR / "SoulX-Singer-Preprocess"


def ensure_pretrained_models():
    """Download SoulX-Singer and Preprocess models from Hugging Face Hub if not present."""
    if (MODEL_DIR_SVS / "model.pt").exists() and (MODEL_DIR_SVS / "model-svc.pt").exists() and MODEL_DIR_PREPROCESS.exists():
        print("Pretrained models already present, skipping download.", flush=True)
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub not installed. Install with: pip install huggingface_hub",
            file=sys.stderr,
            flush=True,
        )
        raise

    PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR_SVS / "model.pt").exists() or not (MODEL_DIR_SVS / "model-svc.pt").exists():
        print("Downloading SoulX-Singer model...", flush=True)
        snapshot_download(
            repo_id="Soul-AILab/SoulX-Singer",
            local_dir=str(MODEL_DIR_SVS),
            local_dir_use_symlinks=False,
        )
        print("SoulX-Singer model ready.", flush=True)

    if not MODEL_DIR_PREPROCESS.exists():
        print("Downloading SoulX-Singer-Preprocess models...", flush=True)
        snapshot_download(
            repo_id="Soul-AILab/SoulX-Singer-Preprocess",
            local_dir=str(MODEL_DIR_PREPROCESS),
            local_dir_use_symlinks=False,
        )
        print("SoulX-Singer-Preprocess models ready.", flush=True)
