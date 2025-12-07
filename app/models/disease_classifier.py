from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import models, transforms
from dataclasses import dataclass, field
from typing import List, Dict, Optional


from app.repositories.base_repository import BaseRepository
from app.core.config import Config


@dataclass
class DiseasePrediction:
    label: str
    probability: float

@dataclass
class DiseaseResult:
    top1: DiseasePrediction
    top2: Optional[DiseasePrediction] = None
    top3: Optional[DiseasePrediction] = None
    raw_probs: Dict[str, float] = field(default_factory=dict)

class DiseaseClassifier(BaseRepository):
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE

        # 기본 클래스 (메타데이터 로딩 실패 시 fallback)
        self.classes = ["건선", "아토피", "여드름", "주사", "지루", "정상"]

        # 전처리용 변수 - Padding Color
        self.pad_color = (124, 116, 104)

        # Transform (Normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:

        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, len(self.classes))

        loading_result = self._load_checkpoint_safe(
            type="FSDD",
            model=model,
            path=self.config.FSDD_CKPT,
            device=self.device
        )

        # Meta Data에서 Class 정보 업데이트
        meta = loading_result.get("meta", {})
        if meta and "classes" in meta:
            self.classes = meta["classes"]

        return loading_result["model"]

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        
        """
        OpenCV 이미지를 PIL로 변환 후 1024x1024 리사이징 & 패딩
        """

        # CV2 BGR -> PIL RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        w, h = img_pil.size
        long_side = max(w, h)
        scale = 1024 / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img_resized = img_pil.resize((new_w, new_h), resample=Image.BICUBIC)

        # 1024x1024 캔버스 생성 (Padding)
        canvas = Image.new("RGB", (1024, 1024), color=self.pad_color)
        left = (1024 - new_w) // 2
        top = (1024 - new_h) // 2
        canvas.paste(img_resized, (left, top))

        # Tensor 변환 및 배치 차원 추가
        return self.transform(canvas).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def classify(self, face_crop_img: np.ndarray) -> DiseaseResult:
        
        """
        Args:
            face_crop_img: 얼굴 크롭 이미지 (FaceDetector 결과)
        """

        input_tensor = self._preprocess(face_crop_img)

        # 추론
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        # top-3 추출
        k = min(3, len(self.classes))
        topk_probs, topk_idxs = probs.topk(k=k, dim=1)

        topk_probs = topk_probs.cpu().numpy().flatten()
        topk_idxs = topk_idxs.cpu().numpy().flatten()

        # 결과 객체 생성
        preds = []
        for i in range(k):
            label = self.classes[topk_idxs[i]]
            prob = float(topk_probs[i])
            preds.append(DiseasePrediction(label, prob))

        raw_map = {self.classes[i]: float(probs[0][i]) for i in range(len(self.classes))}

        return DiseaseResult(
            top1=preds[0],
            top2=preds[1] if k > 1 else None,
            top3=preds[2] if k > 2 else None,
            raw_probs=raw_map
        )