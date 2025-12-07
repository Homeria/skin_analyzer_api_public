from typing import List, Dict, Any
import numpy as np

from app.models.face_detector import FaceDetector
from app.models.part_detector import PartDetector
from app.models.demographics_estimator import DemographicsEstimator
from app.models.disease_classifier import DiseaseClassifier
from app.models.diagnosis_estimator import DiagnosisEstimator
from app.repositories.result_repository import ResultRepository
from app.core.config import Config

from app.utils.image_utils import (
    crop_square_224, mosaic_eyes_with_mediapipe_bgr,
    is_allowed_for_part, canonical_metric_for_pkl, black_ratio,
    PART_ID_TO_STATS, STATS_TO_LABEL
)
from app.utils.ref_stats import RefStats


class SkinAnalysisService:
    def __init__(self,
                 config: Config, 
                 face_detector: FaceDetector,
                 part_detector: PartDetector,
                 demo_estimator: DemographicsEstimator,
                 disease_classifier: DiseaseClassifier,
                 diagnosis_estimator: DiagnosisEstimator,
                 ref_stats: RefStats,
                 result_repo: ResultRepository):

        self.config = config
        
        self.face_detector = face_detector
        self.part_detector = part_detector
        self.demo_estimator = demo_estimator
        self.disease_classifier = disease_classifier
        self.diagnosis_estimator = diagnosis_estimator

        self.ref_stats = ref_stats

        self.result_repo = result_repo

    def analyze(self, img: np.ndarray, real_age: int, real_gender: str):

        # 1. 이미지 로드
        #orig_img = imread_unicode(img_path)
        orig_img = img
        mosaic_img, _ = mosaic_eyes_with_mediapipe_bgr(orig_img.copy())

        # 2, 얼굴 탐지와 크롭
        face_result = self.face_detector.detect(mosaic_img)
        face_image = face_result.face_image

        # 3. 부위 좌표 탐지
        part_result = self.part_detector.detect(face_image)
        
        # 4. 성별/나이 분석
        demo_result = self.demo_estimator.estimate(face_image)

        # 5. 질환 분류
        disease_result = self.disease_classifier.classify(face_image)

        print(disease_result)

        # 6. 부위별 이미지 크롭 (좌표 -> 이미지 리스트)
        parts_crops = []
        ordered_boxes = []

        for i in range(1, 9): # 1~8번 부위
            part_name = PART_ID_TO_STATS[i]
            box = part_result.get_box(part_name) # (x, y, w, h)
            
            # 224x224 규격으로 안전하게 크롭
            crop = crop_square_224(face_image, box, margin=0.12)
            parts_crops.append(crop)
            ordered_boxes.append(box)

        # 7. 상세 진단 (순회 및 통계 조회 로직 포함)
        diagnosis_details = self._process_diagnosis_details(
            parts_crops=parts_crops,
            boxes=ordered_boxes,
            gender=real_gender,
            age=real_age
        )

        # 8. 최종 결과 조립
        final_result = {
            "meta": {
                "gender": demo_result.gender,
                "age": demo_result.age,
                "disease": {
                    "top1": {
                        "label" : disease_result.top1.label,
                        "prob": disease_result.top1.probability
                    },
                    "top2": {
                        "label" : disease_result.top2.label,
                        "prob": disease_result.top2.probability
                    },
                    "top3": {
                        "label" : disease_result.top3.label,
                        "prob": disease_result.top3.probability
                    }
                }
            },
            "parts_analysis": diagnosis_details
        }

        # 9. 결과 저장 (디버깅용)
        self.result_repo.save_all(
            orig_img = orig_img,
            mosaic_img=mosaic_img,
            face_img=face_image,
            parts_crops=parts_crops,
            json_result=final_result,
            boxes=ordered_boxes
        )

        return final_result

        

    def _process_diagnosis_details(self, 
                                   parts_crops: List[np.ndarray], 
                                   boxes: List[tuple], 
                                   gender: str, 
                                   age: int) -> List[Dict[str, Any]]:
        """
        8개 부위를 순회하며 모델 결과(Z-score)를 수치 및 등급으로 변환
        """
        print(f"real_age = {age}")
        # A. 모델 추론 (한 번에 8개 배치 처리)
        raw_res = self.diagnosis_estimator.estimate(parts_crops)
        
        # B. 모델 메타데이터 가져오기 (Denormalization용)
        reg_keys = self.diagnosis_estimator.reg_keys
        reg_mean = self.diagnosis_estimator.reg_mean
        reg_std = self.diagnosis_estimator.reg_std
        
        final_regions = []

        # C. 8개 부위 순회 (Loop)
        for i in range(8):
            stats_name = PART_ID_TO_STATS[i + 1] # e.g., "forehead"
            label_name = STATS_TO_LABEL[stats_name] # e.g., "이마" or "forehead"
            
            # 결과 담을 그릇
            reg_map = {}   # 수치 (Denormalized)
            pct_map = {}   # 백분위 (Percentile)
            grd_map = {}   # 등급 (Grade)
            cls_map = {}   # 분류 등급 (Class)

            # --- (1) Regression (주름, 색소 등 연속값) 처리 ---
            if raw_res.reg_z_scores is not None:
                # 모델이 출력한 모든 Key(주름, 수분 등)를 확인
                for j, key_raw in enumerate(reg_keys):
                    # 현재 부위에 해당하지 않는 항목은 스킵 (예: 이마에서 팔자주름 측정 X)
                    #
                    if not is_allowed_for_part(stats_name, key_raw, is_cls=False):
                        continue
                    
                    # 1. 값 복원 (Z-score -> Real Value)
                    z_val = float(raw_res.reg_z_scores[i, j])
                    mu = reg_mean.get(key_raw, 0.0)
                    sd = reg_std.get(key_raw, 1.0)
                    
                    # 표준편차가 너무 작으면 Z값 그대로 사용
                    real_val = z_val * sd + mu if sd > 1e-12 else z_val
                    reg_map[key_raw] = float(real_val)

                    # 2. 통계 점수 조회 (RefStats)
                    # 통계 파일(pickle) 검색을 위한 표준 이름 변환
                    metric_name = canonical_metric_for_pkl(stats_name, key_raw)
                    
                    stats_score = self.ref_stats.lookup(
                        sex=gender,
                        age=age,
                        region=stats_name, # "forehead"
                        metric=metric_name, # "wrinkle"
                        value=real_val,
                        reg_mean=reg_mean,
                        reg_std=reg_std,
                        fallback_key=key_raw
                    )
                    
                    pct_map[metric_name] = stats_score["percentile"]
                    grd_map[metric_name] = stats_score["grade"]

            # --- (2) Classification (여드름 단계 등 이산값) 처리 ---
            for k, indices in raw_res.cls_indices.items():
                if not is_allowed_for_part(stats_name, k, is_cls=True):
                    continue
                # 0부터 시작하는 인덱스를 1등급부터 시작하도록 변환 (+1)
                cls_map[k] = int(indices[i]) + 1

            # --- (3) 이미지 품질 체크 ---
            q_score = black_ratio(parts_crops[i])

            # 결과 리스트에 추가
            final_regions.append({
                "part_id": i + 1,
                "part_name": label_name,    # "forehead"
                "bbox": boxes[i],           # (x,y,w,h)
                "measurements": reg_map,    # { "wrinkle_forehead": 0.52, ... }
                "percentiles": pct_map,     # { "wrinkle": 45.2, ... }
                "grades": grd_map,          # { "wrinkle": "B+", ... }
                "classes": cls_map,         # { "acne_grade": 1 }
                "quality_score": q_score
            })
            
        return final_regions

    
        
# if __name__ == "__main__":
#     config = Config()
#     service = SkinAnalysisService(config)
#     result = service.analyze(config.TEST_IMG)