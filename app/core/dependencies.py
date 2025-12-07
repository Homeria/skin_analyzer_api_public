# core/dependencies.py
from fastapi import Depends
from functools import lru_cache
from app.services.skin_analysis_service import SkinAnalysisService
from app.models.face_detector import FaceDetector
from app.models.part_detector import PartDetector
from app.models.demographics_estimator import DemographicsEstimator
from app.models.disease_classifier import DiseaseClassifier
from app.models.diagnosis_estimator import DiagnosisEstimator
from app.utils.ref_stats import RefStats
from app.repositories.result_repository import ResultRepository
from app.core.config import Config
from app.core.config2 import Config2

@lru_cache
def get_config() -> Config:
    """
    Config의 싱글톤 인스턴스 반환
    """
    return Config()

@lru_cache
def get_config2() -> Config2:
    """
    Config2의 싱글톤 인스턴스 반환
    """
    return Config2()

@lru_cache
def get_face_detector() -> FaceDetector:
    """
    FaceDetector의 싱글톤 인스턴스 반환
    """
    return FaceDetector(get_config())

@lru_cache
def get_part_detector() -> PartDetector:
    """
    PartDetector의 싱글톤 인스턴스 반환
    """
    return PartDetector(get_config())

@lru_cache
def get_demographics_estimator() -> DemographicsEstimator:
    """
    DemographicsEstimator의 싱글톤 인스턴스 반환
    """
    return DemographicsEstimator(get_config())

@lru_cache
def get_disease_classifier() -> DiseaseClassifier:
    """
    DiseaseClassifier의 싱글톤 인스턴스 반환
    """
    return DiseaseClassifier(get_config())

@lru_cache
def get_diagnosis_estimator() -> DiagnosisEstimator:
    """
    DiagnosisEstimator의 싱글톤 인스턴스 반환
    """
    return DiagnosisEstimator(get_config())

@lru_cache
def get_ref_stats() -> RefStats:
    """
    RefStat의 싱글톤 인스턴스 반환
    """
    return RefStats(get_config())

@lru_cache
def get_result_repository() -> ResultRepository:
    """
    ResultRepository의 싱글톤 인스턴스 반환
    """
    return ResultRepository(get_config())



def get_skin_analysis_service(
        face_detector = Depends(get_face_detector),
        part_detector = Depends(get_part_detector),
        demographics_estimator = Depends(get_demographics_estimator),
        disease_classifier = Depends(get_disease_classifier),
        diagnosis_estimator = Depends(get_diagnosis_estimator),
        ref_stats = Depends(get_ref_stats),
        result_repository = Depends(get_result_repository)
    ) -> SkinAnalysisService:
        return SkinAnalysisService(
            config=get_config(),
            face_detector=face_detector,
            part_detector=part_detector,
            demo_estimator=demographics_estimator,
            disease_classifier=disease_classifier,
            diagnosis_estimator=diagnosis_estimator,
            ref_stats=ref_stats,
            result_repo=result_repository
        )

