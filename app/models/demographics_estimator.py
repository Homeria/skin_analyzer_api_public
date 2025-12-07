from __future__ import annotations
import torch
import torch.nn as nn 
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict
from torchvision.models import convnext_tiny    

from app.core.config import Config
from app.repositories.base_repository import BaseRepository
from app.utils.image_utils import (
    bgr_to_norm_tensor, decode_gender_age_from_demo
)


@dataclass
class DemographicsResult:
    gender: str
    age: int

class DemoModel(nn.Module):
    
    """
    성별(2 class - M/F)과 나이(1 scalar)를 동시에 추론하는 모델
    """

    def __init__(self):
        super().__init__()
        
        # Backbone
        b = convnext_tiny(weights=None)
        dim = b.classifier[2].in_features
        b.classifier[2] = nn.Identity()
        self.backbone = b

        # Two Heads
        self.fc_gender = nn.Linear(dim, 2)
        self.fc_age = nn.Linear(dim, 1)

    def forward(self, x):
        f = self.backbone(x)
        return {"gender": self.fc_gender(f), "age": self.fc_age(f).squeeze(1)}


class DemographicsEstimator(BaseRepository):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE
        self.model = self._load_model_custom()

    def _load_model_custom(self) -> nn.Module:
        """
        
        """
        model = DemoModel().to(self.device)
        path = self.config.DEMO_CKPT

        if not path.exists():
            raise FileNotFoundError(f"[DEMO] Checkpoint not found: {path}")
        
        ckpt = torch.load(str(path), map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        renamed = {}
        for k, v, in sd.items():
            # module. 접두어 제거
            nk = k[7:] if k.startswith("module.") else k
            renamed[nk] = v

        def find_key(cands, source_dict):
            for c in cands:
                if c in source_dict: return c
            for k in source_dict.keys():
                for c in cands:
                    if c in k: return k
            return None
        
        fixed = dict(renamed)

        # Gender Head 매핑
        g_w = find_key(["fc_gender.weight", "gender_head.weight", "head_gender.weight", "classifier_gender.weight"], renamed)
        g_b = find_key(["fc_gender.bias", "gender_head.bias"], renamed)
        if g_w: fixed["fc_gender.weight"] = renamed[g_w]
        if g_b: fixed["fc_gender.bias"] = renamed[g_b]

        # Age Head 매핑
        a_w = find_key(["fc_age.weight", "age_head.weight", "head_age.weight"], renamed)
        a_b = find_key(["fc_age.bias", "age_head.bias"], renamed)
        if a_w: fixed["fc_age.weight"] = renamed[a_w]
        if a_b: fixed["fc_age.bias"] = renamed[a_b]

        # 로딩 (strict=False로 불일치 허용하되, 핵심 키는 위에서 맞춤)
        model.load_state_dict(fixed, strict=False)
        model.eval()
        
        return model

    def estimate(self, face_crop_img: np.ndarray) -> DemographicsResult:

        """
        Args:
            face_crop_img: FaceDetector에서 나온 960x720 (또는 유사비율) 얼굴 이미지
        """

        tensor_input = bgr_to_norm_tensor(face_crop_img, self.device)
        out = self.model(tensor_input)
        gender, age = decode_gender_age_from_demo(out)

        return DemographicsResult(gender=gender, age=age)


