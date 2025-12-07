from app.core.exceptions import *
from fastapi import UploadFile
from typing import List

def validate_request(
        image: UploadFile,
        gender: str = None,
        birth_year: int = None,
        birth_month: int = None,
        concerns: List[str] = None
) -> bool:
    pass