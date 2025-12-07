# validators/analysis_validator.py
import numpy as np
from app.core.exceptions import FaceNotFoundError, FaceOccludedError

class AnalysisValidator:

    """
    AI 모델의 전처리 및 분석 결과를 검증하는 클래스
    """

    def validate_crop_result(self, crop_result: dict | None):
        """
        ImageProcessRepository의 결과(crop_result)가 유효한지 검사.
        """
        
        # 1. 얼굴 감지 실패 (None 반환된 경우)
        if crop_result is None:
            raise FaceNotFoundError()

        # 2. 주요 부위 가림(Occlusion) 검사
        # parts_xywh 리스트에 [0, 0, 0, 0] 같은 빈 좌표가 있는지 확인
        parts = crop_result.get("parts_xywh", [])

        # 인덱스 별 부위 이름 (에러 메시지용)
        part_names = ["이마", "미간", "오른쪽 눈", "왼쪽 눈", "오른쪽 볼", "왼쪽 볼", "입술", "턱"]

        for i, bbox in enumerate(parts):
            # bbox가 [0, 0, 0, 0] 이거나 너비/높이가 너무 작으면 가려진 것으로 판단
            w, h = bbox[2], bbox[3]
            if w < 5 or h < 5: 
                raise FaceOccludedError(part_name=part_names[i])