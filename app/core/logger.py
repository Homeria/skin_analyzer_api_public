import logging
import json
import os
import cv2
import numpy as np
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List

# ==========================================
# 1. JSON Encoder (Numpy 호환)
# ==========================================
class NumpyEncoder(json.JSONEncoder):
    """Numpy 데이터 타입(int64, float32, ndarray)을 JSON으로 변환하기 위한 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# ==========================================
# 2. System Logger (서버 로그용)
# ==========================================
def setup_system_logger(log_file: str = "server.log"):
    """
    애플리케이션 전역 로거 설정 (서버 시작 시 1회 호출 권장)
    """
    logger = logging.getLogger("skin_analyzer_api")
    logger.setLevel(logging.INFO)
    
    # 중복 핸들러 방지
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s"
    )

    # 1. 콘솔 출력 (터미널)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. 파일 출력
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 전역에서 쓸 수 있는 시스템 로거 인스턴스
sys_logger = setup_system_logger()


# ==========================================
# 3. Artifact Logger (AI 결과물 저장용)
# ==========================================
class ArtifactLogger:
    """
    요청 1건에 대한 이미지, JSON, 텍스트 결과를 저장하는 로거.
    FastAPI의 Depends로 주입받아 사용.
    """
    def __init__(self, base_dir: str = "debug_images", enable: bool = True):
        self.enable = enable
        if not self.enable:
            return

        # 고유 ID 생성 (시간 + 랜덤 UUID 앞자리) -> 폴더명 중복 절대 방지
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        self.request_id = f"{timestamp}_{unique_id}"
        
        # 저장 경로: debug_images/20251206_120000_a1b2c3d4/
        self.save_dir = Path(base_dir) / self.request_id
        
        # (옵션) 생성 시점에 폴더를 만들지 않고, 실제 저장할 때 만들려면 여기를 주석 처리
        self.ensure_dir()

    def ensure_dir(self):
        """디렉토리가 없으면 생성"""
        if self.enable and not self.save_dir.exists():
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def save_image(self, image: np.ndarray, filename: str, sub_dir: str = ""):
        """
        이미지를 저장합니다.
        :param image: Numpy 이미지 배열 (BGR)
        :param filename: 파일명 (예: crop_forehead.jpg)
        :param sub_dir: (옵션) 하위 폴더명 (예: parts)
        """
        if not self.enable or image is None or image.size == 0:
            return

        try:
            target_dir = self.save_dir / sub_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = target_dir / filename
            success = cv2.imwrite(str(save_path), image)
            if not success:
                sys_logger.warning(f"이미지 저장 실패 (cv2 error): {save_path}")
                
        except Exception as e:
            sys_logger.error(f"이미지 저장 중 예외 발생: {e}")

    def save_json(self, data: Dict[str, Any], filename: str = "result.json"):
        """결과 데이터를 JSON으로 저장합니다."""
        if not self.enable:
            return

        try:
            save_path = self.save_dir / filename
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        except Exception as e:
            sys_logger.error(f"JSON 저장 실패: {e}")

    def save_text(self, text: str, filename: str = "log.txt"):
        """간단한 텍스트 로그 저장"""
        if not self.enable:
            return
            
        try:
            save_path = self.save_dir / filename
            with open(save_path, "a", encoding="utf-8") as f: # append 모드
                f.write(text + "\n")
        except Exception as e:
            sys_logger.error(f"텍스트 저장 실패: {e}")

    def get_log_dir(self) -> str:
        """현재 로그가 저장되는 경로 반환"""
        return str(self.save_dir) if self.enable else ""