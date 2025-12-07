# main.py
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import numpy as np
from app.api.analysis_controller import router as analysis_router
from app.dtos import RootResponse
from app.core.exceptions import *
from app.core.dependencies import get_skin_analysis_service
from app.core.config import Config


#from .core.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        service = get_skin_analysis_service()
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            service.analyze(dummy_image)
        except Exception:
            pass
    except Exception as e:
        print(f"❌ [Warning] 워밍업 중 문제 발생 (치명적이지 않음): {e}")
    yield


# 1. FastAPI 애플리케이션 생성
app = FastAPI(lifespan=lifespan, title="피부 분석 AI API")

config = Config()


# 2. CORS 미들웨어 설정
origins = [
    config.DEVELOP_URL
]
print(origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_user_agent(request: Request, call_next):
    user_agent = request.headers.get("user-agent")
    print(f"👀 [User-Agent 감지]: {user_agent}")
    response = await call_next(request)
    return response


# 3. API 라우터 등록
app.include_router(analysis_router)

# 4. 루트 경로 (서버 상태 확인용)
@app.get("/", response_model=RootResponse, summary="API 정보 확인")
def read_root():
    """API의 기본 정보와 문서 링크를 제공합니다."""
    return {
        "project_name": "Skin Analyzer AI API",
        "version": "1.0.0",
        "description": "얼굴 이미지를 분석하여 피부 나이와 상태를 예측하는 API입니다.",
        "docs_url": "/docs"
    }

@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """
    요청으로 들어온 데이터 검증 에러에 대한 핸들러
    """

    #logger.warning(f"클라이언트 유효성 검사 실패: {exc.detail} (Path: {request.url.path})")
    
    status_code = status.HTTP_400_BAD_REQUEST # 기본값: 400
    
    # 특별 케이스: '얼굴 없음' 오류는 422로 처리
    if isinstance(exc, FaceNotFoundError):
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT

    if isinstance(exc, APIKeyMismatchError):
        status_code = status.HTTP_403_FORBIDDEN
    
    return JSONResponse(
        status_code=status_code,
        content={"detail": exc.detail, "error_type": type(exc).__name__},
    )

@app.exception_handler(AIException)
async def ai_exception_handler(request: Request, exc: AIException):
    """
    AI 모델 처리 중 발생한 서버 내부 오류에 대한 핸들러
    """

    #logger.error(f"AI 처리 중 오류: {exc.detail}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": type(exc).__name__, # 예: ModelLoadError
            "detail": exc.detail,             # 예: "AI 모델(SkinAge)을 로드하는 중..."
            "path": request.url.path          # 에러가 발생한 API 경로
        },
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):

    #logger.critical(f"예상치 못한 시스템 오류: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_type": type(exc).__name__, # 예: KeyError, ValueError
            "detail": str(exc),               # 예: "'condition_scores'"
            "message": "서버 코드 내부에서 처리되지 않은 버그가 발생했습니다."
        },
    )





