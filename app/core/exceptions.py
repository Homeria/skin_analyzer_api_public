# ==========================================
# 1. 클라이언트 요청 오류 (4xx Bad Request / 422 Unprocessable)
# ==========================================

class ValidationException(Exception):
    """잘못된 요청에 대한 에러의 기본 클래스"""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

# --- 파일 및 형식 관련 (400) ---
class InvalidImageTypeError(ValidationException):
    def __init__(self):
        super().__init__(detail="지원하지 않는 파일 형식입니다. jpg(jpeg), png 파일만 업로드 가능합니다.")

class CorruptedImageError(ValidationException):
    def __init__(self):
        super().__init__(detail="이미지 파일이 손상되어 열 수 없습니다.")

class ImageTooLargeError(ValidationException):
    def __init__(self, max_mb: int = 10):
        super().__init__(detail=f"이미지 용량이 너무 큽니다. (최대 {max_mb}MB)")

class ImageTooSmallError(ValidationException):
    def __init__(self, min_size: int = 224):
        super().__init__(detail=f"이미지 해상도가 너무 낮습니다. 최소 가로/세로 {min_size}px 이상이어야 합니다.")


# --- 잘못된 접근 관련 (403) ---
class APIKeyMismatchError(ValidationException):
    """API 키가 맞지 않을 때"""
    def __init__(self, field: str):
        super().__init__(detail=f"입력한 API Key '{field} 올바르지 않습니다.")

# --- 데이터 내용 관련 (422) ---
class InvalidMetadataError(ValidationException):
    """성별, 나이 등 폼 데이터가 이상할 때"""
    def __init__(self, field: str):
        super().__init__(detail=f"입력된 '{field}' 정보가 올바르지 않습니다.")

# --- AI 전처리/분석 불가 관련 (422) ---
class FaceNotFoundError(ValidationException):
    def __init__(self):
        super().__init__(detail="이미지에서 얼굴을 찾을 수 없습니다. 정면 얼굴 사진을 사용해주세요.")

class MultipleFacesFoundError(ValidationException):
    """얼굴이 너무 많이 잡혀서 누구를 분석할지 모를 때"""
    def __init__(self, count: int):
        super().__init__(detail=f"얼굴이 {count}개 감지되었습니다. 한 명만 나온 독사진을 사용해주세요.")

class FaceOccludedError(ValidationException):
    """마스크, 선글라스, 손 등으로 주요 부위가 가려졌을 때"""
    def __init__(self):
        super().__init__(detail="얼굴의 일부가 가려져 있어 분석할 수 없습니다. 마스크나 안경을 벗고 다시 시도해주세요.")


# ==========================================
# 2. 서버 내부 오류 (5xx Internal Server Error)
# ==========================================

class AIException(Exception):
    """AI 모델 관련 서버 내부 에러의 기본 클래스"""
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)

class ModelLoadError(AIException):
    """모델 파일이 없거나 로딩 중 에러"""
    def __init__(self, model_name: str):
        super().__init__(detail=f"AI 모델({model_name})을 로드하는 중 치명적인 오류가 발생했습니다.")

class AIInferenceError(AIException):
    """추론 도중 알 수 없는 에러 (Tensor shape 불일치 등)"""
    def __init__(self, stage: str):
        super().__init__(detail=f"AI 분석 단계({stage})에서 내부 오류가 발생했습니다.")

class AIOutputNaNError(AIException):
    """모델이 결과값으로 숫자가 아닌 NaN/Infinity를 뱉었을 때"""
    def __init__(self):
        super().__init__(detail="AI 분석 결과가 유효하지 않습니다 (NaN). 다른 이미지로 시도해주세요.")