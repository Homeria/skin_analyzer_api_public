from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import torch.nn as nn
import torch


class BaseRepository:
    def __init__(self):
        pass

    def _load_checkpoint_safe(self, type: str, model: nn.Module, path: Path, device) -> Tuple[nn.Module, dict]:

        if not path.exists():
            raise FileNotFoundError(f"[{type}] Checkpoint is not found: {path}")

        model = model.to(device)
        ckpt = torch.load(str(path), map_location="cpu")
        meta = ckpt if isinstance(ckpt, dict) else {}
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass

        model.eval()
        if isinstance(ckpt, dict):
            meta = ckpt

        return {
            "model" : model,
            "ckpt" : ckpt,
            "meta" : meta,
            "sd" : sd
        }
    
    
