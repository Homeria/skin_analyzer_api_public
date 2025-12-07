from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any
from torchvision.models import convnext_tiny

from app.core.config import Config
from app.repositories.base_repository import BaseRepository
from app.utils.image_utils import (
    bgr_to_norm_tensor
)

@dataclass
class DiagnosisRawResult:
    """모델의 순수 출력값 (후처리 전)"""
    reg_z_scores: np.ndarray  # (8, NumRegKeys) - 정규화된 Z값
    cls_indices: Dict[str, np.ndarray]  # {key: (8,) class_index}

class DiagModel(nn.Module):
    def __init__(self, meta: Dict[str, Any]):
        super().__init__()
        # Backbone
        b = convnext_tiny(weights=None)
        dim = b.classifier[2].in_features
        b.classifier[2] = nn.Identity()
        self.backbone = b

        # Metadata 로드
        self.reg_keys = meta.get("reg_keys", [])
        self.cls_keys = meta.get("cls_keys", [])
        self.cls_nc = meta.get("cls_nc", {})

        # Heads 생성
        # Regression Head
        self.reg_head = nn.Linear(dim, len(self.reg_keys)) if self.reg_keys else None
        
        # Classification Heads (Multi-task)
        self.cls_heads = nn.ModuleDict({
            k: nn.Linear(dim, self.cls_nc.get(k, 2)) for k in self.cls_keys
        })

    def forward(self, x):
        f = self.backbone(x)
        reg = self.reg_head(f) if self.reg_head is not None else None
        cls = {k: head(f) for k, head in self.cls_heads.items()}
        return {"reg": reg, "cls": cls}

class DiagnosisEstimator(BaseRepository):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE
        
        # 메타데이터 저장을 위한 변수
        self.reg_keys = []
        self.cls_keys = []
        self.reg_mean = {}
        self.reg_std = {}
        self.cls_nc = {}

        self.model = self._load_model_with_meta()

    def _load_model_with_meta(self) -> nn.Module:
        path = self.config.DIAG_CKPT
        if not path.exists():
            raise FileNotFoundError(f"[DIAG] Checkpoint not found: {path}")

        # 1. 체크포인트 먼저 로드하여 메타데이터 확보
        ckpt = torch.load(str(path), map_location="cpu")
        
        # 메타데이터 추출 (Self에 저장하여 서비스에서 접근 가능하게 함)
        self.reg_keys = ckpt.get("reg_keys", [])
        self.cls_keys = ckpt.get("cls_keys", [])
        self.reg_mean = ckpt.get("reg_mean", {})
        self.reg_std = ckpt.get("reg_std", {})
        self.cls_nc = ckpt.get("cls_nc", {})

        meta_for_model = {
            "reg_keys": self.reg_keys,
            "cls_keys": self.cls_keys,
            "cls_nc": self.cls_nc
        }

        # 2. 모델 구조 빌드
        model = DiagModel(meta_for_model)

        # 3. 가중치 로드
        # (체크포인트 구조가 dict 통째로 되어있을 수 있으므로 유연하게 처리)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        model.load_state_dict(sd, strict=False)
        
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def estimate(self, crop_images: List[np.ndarray]) -> DiagnosisRawResult:
        """
        Args:
            crop_images: 8개 부위의 크롭 이미지 리스트 (224x224 권장)
        """
        if not crop_images:
            raise ValueError("No crop images provided")

        # 배치 변환 (List[H,W,3] -> Tensor[B,3,H,W])
        tensors = [bgr_to_norm_tensor(img, self.device) for img in crop_images]
        batch_input = torch.cat(tensors, dim=0) # (8, 3, 224, 224)

        # 추론
        out = self.model(batch_input)
        
        # 결과 정리 (CPU Numpy)
        reg_out = out["reg"]
        reg_z = reg_out.detach().cpu().numpy() if reg_out is not None else np.empty((len(crop_images), 0))
        
        cls_res = {}
        for k, logits in out["cls"].items():
            # Argmax로 클래스 인덱스 추출
            indices = torch.argmax(logits, dim=1).detach().cpu().numpy()
            cls_res[k] = indices

        return DiagnosisRawResult(reg_z_scores=reg_z, cls_indices=cls_res)