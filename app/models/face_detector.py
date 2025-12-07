from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Any, Dict, List, Tuple, Optional
from torchvision.models import convnext_tiny
from app.repositories.base_repository import BaseRepository    
from app.utils.image_utils import (
    resize_to_full, bgr_to_norm_tensor, clamp_box,resize_face_to_inner,
    FULL_W, FULL_H

)
import torch
import torch.nn as nn 
from app.core.config import Config
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")

@dataclass
class FaceDetectionResult:
    full_image: np.ndarray
    face_image: np.ndarray
    face_box: Tuple[int, int, int, int]


class FaceDetector(BaseRepository):

    """
    이미지에서 얼굴 영역을 추론하고, 후속 모델이 사용할 수 있도록 크롭/가공하여 반환하는 클래스
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.device = config.DEVICE

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:

        """
        모델 구조 생성 및 체크포인트 로드
        """

        #model = FaceModel()

        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 4)
        
        loading_result = self._load_checkpoint_safe(
            type="FACE",
            model=model,
            path=self.config.FACE_CKPT,
            device=self.device
        )

        return loading_result["model"]
    
    def detect(self, img_bgr: np.ndarray) -> FaceDetectionResult:

        """
        메인 추론 메서드
        Args:
            img_bgr: 원본 입력 이미지 (OpenCV BGR)
        Returns:
            FaceDetectionResult: 구조화된 결과 데이터
        """

        # [전처리]

        # 1. 전체 이미지 리사이즈 (1440x1080)
        # 반환값 : 이미지, 스케일, 패딩좌표(L, T)
        full_resized, _, _, _ = resize_to_full(img_bgr)

        # 2. 텐서 변환 (Normalize 포함)
        tensor_input = bgr_to_norm_tensor(full_resized, self.device)

        # [추론]

        # 3. 모델 실행
        pred = self.model(tensor_input).detach().cpu().numpy().squeeze(0).astype(float)

        # [후처리]

        # 4. 좌표 보정 (Clamp)
        fx, fy, fw, fh = pred.tolist()
        fx1, fy1, fx2, fy2 = clamp_box(fx, fy, fw, fh, FULL_W, FULL_H)

        # 5. 얼굴 영역 크롭
        face_crop = full_resized[fy1:fy2, fx1:fx2]

        # 얼굴이 인식되지 않음
        if face_crop.size == 0:
            raise RuntimeError("Face Detection Failed : Empty Face Crop") 

        # 6. 다음 모델(Part, Demo) 입력을 위한 리사이즈 (1440x1080 -> 960x720)
        # resize_face_to_inner는 내부에서 패팅 처리를 수행
        face_crop_resized, _, _, _ = resize_face_to_inner(face_crop)

        return FaceDetectionResult(
            full_image=full_resized,
            face_image=face_crop_resized,
            face_box=(fx1, fy1, fx2, fy2)
        )