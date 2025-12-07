# api/analysis_controller.py
from fastapi import APIRouter, File, UploadFile, Depends, Form, Header
from typing import List
from app.services.skin_analysis_service import SkinAnalysisService
from app.dtos import SkinAnalysisResult, ConfigResponse
from app.core.dependencies import (
    get_skin_analysis_service,
    get_config
)
from app.core.config import Config
#from app.core.logger import PredictionLogger
from app.utils.image_utils import (
    decode_image_bytes
)
from app.validators.request_validator import RequestValidator
from app.core.exceptions import *

# API 라우터 생성
router = APIRouter()

@router.post("/analyze/skin", response_model=SkinAnalysisResult, summary="피부 이미지 분석")
async def analyze_skin(
    api_key: str = Header(..., alias="x-api-key", description="고유 API 키"),
    image: UploadFile = File(..., description="분석할 얼굴 이미지 파일"),
    gender: str = Form(..., description="성별 (예: 여성)"),
    birth_year: int = Form(..., description="출생년도 (예: 1995)"),
    birth_month: int = Form(..., description="출생월 (예: 10)"),
    concerns: List[str] = Form(..., description="피부 고민 목록 (예: ['주름', '건조'])"),

    config: Config = Depends(get_config),
    service: SkinAnalysisService = Depends(get_skin_analysis_service)
):
    """
    사용자 정보(성별, 나이, 고민)와 얼굴 이미지를 함께 받아 피부 분석 파이프라인을 실행하고 결과를 반환.
    """

    contents = await image.read()

    request_validator = RequestValidator()
    request_validator.validate(config, api_key, image, contents, gender, birth_year, birth_month, concerns)

    #logger.info(f"분석 요청 수신 : 성별-{gender}, 출생-{birth_year}년 {birth_month}월, 고민-{concerns}")

    config.init_request()
    
    try:
        image_matrix = decode_image_bytes(contents)
    except ValueError:
        raise ValueError
    
    result = service.analyze(image_matrix, (2025 - birth_year), gender)

    return result

@router.get("/health", summary="서버 상태 확인")
def health_check():

    """서버가 정상적으로 동작하는지 확인합니다."""
    return {"status": "ok"}

@router.get("/config", response_model=ConfigResponse, summary="분석 설정 정보")
def get_config():
    """현재 AI 모델의 버전과 분석 가능한 피부 상태 목록을 반환합니다."""
    return {
        "model_version": "1.0.0",
        "available_conditions": ["Psoriasis", "Atopic Dermatitis", "Acne", "Rosacea", "Normal"]
    }