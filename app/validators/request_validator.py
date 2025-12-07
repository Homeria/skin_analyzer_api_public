# validators/request_validator.py
from fastapi import UploadFile, File
from PIL import Image
import io
import numpy as np
from typing import List
from app.core.exceptions import (
    InvalidImageTypeError,
    ImageTooSmallError,
    InvalidMetadataError,
    APIKeyMismatchError
)
from app.core.config import Config

class RequestValidator:

    """
    업로드된 이미지 파일과 메타데이터(성별, 나이, 고민 등)를 검증하는 클래스
    """
    
    def __init__(self):
        pass
    
    def validate(self,
                 config: Config,
                 api_key: str,
                 image: UploadFile,
                 image_bytes: bytes,
                 gender: str,
                 birth_year: str,
                 birth_month: str,
                 concerns: List[str]):
        
        ALLOWED_MIMES = ["image/jpeg", "image/png"]

        if config.API_KEY != api_key:
            raise APIKeyMismatchError(api_key)
        
        if image.content_type not in ALLOWED_MIMES:
            raise InvalidImageTypeError()
        
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil_image.size
        if w < 224 or h < 224:
            raise ImageTooSmallError(min_size=224)
        
        if gender and gender not in ["M", "F", "m", "f", "남성", "여성"]:
             raise InvalidMetadataError("성별(gender)은 'M' 또는 'F'여야 합니다.")

        if birth_year:
            if birth_year < 1900 or birth_year > 2025:
                raise InvalidMetadataError("올바르지 않은 출생년도입니다.")
            
        if birth_month:
            if birth_month < 1 or birth_month > 12:
                raise InvalidMetadataError("출생월(birth_month)은 1에서 12 사이여야 합니다.")

        if concerns:
            valid_concerns = ["주름", "칙칙함", "기미/잡티", "모공", "피지 과다", "민감성", "탄력저하", "각질", "다크서클", "건조", "여드름", "홍조"]
            for c in concerns:
                if c not in valid_concerns:
                    raise InvalidMetadataError(f"지원하지 않는 피부 고민입니다: {c}")