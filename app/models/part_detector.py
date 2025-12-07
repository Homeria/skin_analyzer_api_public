from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
from torchvision.models import convnext_tiny

from app.core.config import Config  
from app.repositories.base_repository import BaseRepository  
from app.utils.image_utils import (
    bgr_to_norm_tensor, clamp_box,
    FACE_W, FACE_H, PART_ID_TO_STATS
)


@dataclass
class PartDetectionResult:
    """
    부위별 탐지 결과
    parts: { "forehead" : (x, y, w, h), "left_eyes" : (x, y, w, h), ...}
    ordered_boxes : 모델이 출력한 순서대로 정렬된 리스트
    """
    parts: Dict[str, Tuple[int, int, int, int]] = field(default_factory=dict)
    ordered_boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)

    def get_box(self, part_name: str) -> Tuple[int, int, int, int]:
        return self.parts.get(part_name, (0, 0, 0, 0))


class PartDetector(BaseRepository):

    """
    얼굴 크롭 이미지 내에서 8개의 주요 부위의 좌표(Bounding Box)를 탐지하여 값을 반환하는 클래스
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

        # model = PartModel(num_parts=8)
        
        num_parts = 8
        model = convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_parts * 4)

        loading_result = self._load_checkpoint_safe(
            type="PART",
            model=model,
            path=self.config.PART_CKPT,
            device=self.device
        )

        return loading_result["model"]
    
    @torch.no_grad()
    def detect(self, face_crop_img: np.ndarray) -> PartDetectionResult:
        """
        메인 추론 메서드
        Args:
            face_crop_img: FaceDetector에서 넘어온 960x720 얼굴 크롭 이미지
        Returns:
            PartDetectionResult: 부위별 좌표 맵
        """

        # [전처리]

        # 1. 텐서 변환
        # 모델 입력 크기에 맞춰 Normalize 수행
        tensor_input = bgr_to_norm_tensor(face_crop_img, self.device)
        
        # [추론]

        # 2. 추론 진행
        # output shape: (1, 32) -> (32, )
        pred = self.model(tensor_input).detach().cpu().numpy().squeeze(0).astype(float)

        # 아웃풋 미스매치
        if pred.size != 32:
            raise RuntimeError(f"Part detector output mismatch. Expected 32, got {pred.size}")
        
        # [후처리]

        # 3. 결과 파싱
        # (8 parts, 4 coords)
        pred = pred.reshape(8, 4)

        result_map = {}
        ordered_list = []

        for i in range(8):
            x, y, w, h = pred[i].tolist()

            # 4. 좌표 보정 (이미지 밖으로 나가지 않게 clamp)
            # 좌표 기준은 face_crop_img (FACE_W, FACE_H)
            x1, y1, x2, y2 = clamp_box(x, y, w, h, FACE_W, FACE_H)

            # xyxy -> xywh 형태로 변환하여 저장
            final_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # ID 매핑
            part_name = PART_ID_TO_STATS.get(i + 1, f"unknown_{i}")

            result_map[part_name] = final_box
            ordered_list.append(final_box)

        return PartDetectionResult(
            parts=result_map,
            ordered_boxes=ordered_list
        )
    
    

    