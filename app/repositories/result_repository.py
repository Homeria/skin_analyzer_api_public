import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

from app.core.config import Config
from app.utils.image_utils import (
    imwrite_unicode,
    PART_ID_TO_STATS
)

class ResultRepository:
    def __init__(self, config: Config):
        self.config = config

    def save_all(self, 
                 orig_img: np.ndarray, 
                 mosaic_img: np.ndarray,
                 face_img: np.ndarray, 
                 parts_crops: List[np.ndarray], 
                 json_result: Dict[str, Any],
                 boxes: List[tuple] = None):
        """
        모든 디버깅용 데이터 저장
        """

        self.out_dir = self.config.OUTPUT_DIR
        self.parts_dir = self.config.DIR_PARTS

        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.parts_dir.exists():
            self.parts_dir.mkdir(parents=True, exist_ok=True)

        print(f"[ResultRepo] Saving results to: {self.out_dir}")

        # 1. 원본 이미지 저장
        imwrite_unicode(self.out_dir / "00_original_raw.jpg", orig_img)

        # 2. 모자이크된 이미지 저장
        imwrite_unicode(self.out_dir / "01_mosaic.jpg", mosaic_img)

        # 2. 얼굴 크롭 이미지 저장 (박스 그려서 저장하면 더 좋음)
        face_debug = face_img.copy()
        if boxes:
            for i, box in enumerate(boxes):
                x, y, w, h = box
                # 노란색 박스 그리기
                cv2.rectangle(face_debug, (x, y), (x+w, y+h), (0, 255, 255), 2)
                # 부위 이름 쓰기
                name = PART_ID_TO_STATS.get(i+1, str(i))
                cv2.putText(face_debug, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        imwrite_unicode(self.out_dir / "02_face_crop.jpg", face_img)       # 깨끗한 얼굴
        imwrite_unicode(self.out_dir / "02_face_debug.jpg", face_debug)    # 박스 그려진 얼굴

        # 3. 부위별 조각 이미지 저장 (parts 폴더)
        for i, crop in enumerate(parts_crops):
            part_name = PART_ID_TO_STATS.get(i+1, f"part_{i+1}")
            filename = f"{i+1:02d}_{part_name}.jpg"
            imwrite_unicode(self.parts_dir / filename, crop)

        # 4. 결과 JSON 저장
        json_path = self.out_dir / "result.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2, default=str)
        
        print("[ResultRepo] Save Complete.")