from fastapi import UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# 1. 질병 상세 정보 (label, prob를 가진 객체)
class DiseaseDetail(BaseModel):
    label: str = Field(..., description="질병명")
    prob: float = Field(..., description="확률")

# 2. 질병 정보 (top1, top2, top3 객체를 포함)
class DiseaseInfo(BaseModel):
    top1: DiseaseDetail = Field(..., description="1순위 질병")
    top2: Optional[DiseaseDetail] = Field(None, description="2순위 질병")
    top3: Optional[DiseaseDetail] = Field(None, description="3순위 질병")

# 3. 메타 정보
class MetaData(BaseModel):
    gender: str = Field(..., description="성별")
    age: int = Field(..., description="나이")
    disease: DiseaseInfo = Field(..., description="질병 분석 결과")

# 4. 부위별 분석 정보
class PartAnalysis(BaseModel):
    part_id: int = Field(..., description="부위 ID")
    part_name: str = Field(..., description="부위 이름")
    
    bbox: List[int] = Field(..., description="좌표 [x, y, w, h]")

    # 동적 키 값 처리를 위한 Dict 설정 (빈 값 허용)
    measurements: Dict[str, float] = Field(default_factory=dict)
    percentiles: Dict[str, float] = Field(default_factory=dict)
    grades: Dict[str, str] = Field(default_factory=dict)
    classes: Dict[str, int] = Field(default_factory=dict)

    quality_score: float = Field(..., description="이미지 품질 점수")

# 5. 최종 결과
class SkinAnalysisResult(BaseModel):
    meta: MetaData
    parts_analysis: List[PartAnalysis]


class SkinAnalysisResult(BaseModel):
    meta: MetaData
    parts_analysis: List[PartAnalysis]

class ConfigResponse(BaseModel):
    model_version: str
    available_conditions: List[str]

class RootResponse(BaseModel):
    project_name: str = Field(..., example="Skin Analyzer AI API")
    version: str = Field(..., example="1.0.0")
    description: str = Field(..., example="얼굴 이미지를 분석하여 피부 나이와 상태를 예측하는 API입니다.")
    docs_url: str = Field(..., example="/docs")