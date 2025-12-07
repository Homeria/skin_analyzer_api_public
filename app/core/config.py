import torch
from pathlib import Path
from datetime import datetime
import contextvars  # [추가 1] 마법 주머니 임포트
from dotenv import load_dotenv
import os

# [추가 2] 요청(스레드)마다 경로를 따로 저장할 전역 저장소 선언
# 기본값은 서버 켜질 때의 data/default 로 설정 (안전을 위해)
_output_dir_ctx = contextvars.ContextVar("output_dir", default=Path("./data/default"))
_parts_dir_ctx = contextvars.ContextVar("parts_dir", default=Path("./data/default/parts"))

class Config():

    def __init__(self):
        # --- [고정 경로] 서버 켜질 때 한 번만 계산됨 ---
        self.ROOT = Path(__file__).resolve().parent.parent.parent
        self.DATA_DIR = self.ROOT / "data"
        self.APP = self.ROOT / "app"
        
        # 모델 경로들은 변하지 않으므로 __init__에 둠
        self.MODEL = self.APP / "models"
        self.CHECKPOINT = self.MODEL / "checkpoints"
        self.FACE_CKPT = self.CHECKPOINT / "FACE_CKPT.pth"
        self.PART_CKPT = self.CHECKPOINT / "PART_CKPT.pth"
        self.DEMO_CKPT = self.CHECKPOINT / "DEMO_CKPT.pt"
        self.DIAG_CKPT = self.CHECKPOINT / "DIAG_CKPT.pt"
        self.FSDD_CKPT = self.CHECKPOINT / "FSDD_CKPT.pt"
        self.REF_PKL = self.APP / "ref_stats" / "percentiles.pkl"
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TEST_IMG = self.MODEL / "test_img" / "여드름_2.png"

        load_dotenv()
        self.API_KEY = os.getenv("API_KEY")
        self.DEVELOP_URL = os.getenv("DEVELOP_URL")
        self.FRONTEND_URL = os.getenv("FRONTEND_URL")

        # [삭제] TIMESTAMP, OUTPUT_DIR 등은 여기서 계산하면 안 됨! (서버 켤 때 고정되어버림)
        # self.TIMESTAMP = ... (삭제)
        # self.OUTPUT_DIR = ... (삭제)

    # --- [추가 3] 컨트롤러가 호출할 '시간 갱신 및 폴더 생성' 함수 ---
    def init_request(self):
        """
        [Controller용] 요청이 들어올 때마다 호출.
        현재 시간으로 폴더를 만들고 ContextVar에 저장함.
        """
        # 1. 현재 시간 구하기
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 2. 경로 계산
        request_output_dir = self.DATA_DIR / now_str
        request_parts_dir = request_output_dir / "parts"

        # 3. 폴더 생성 (mkdir)
        request_output_dir.mkdir(parents=True, exist_ok=True)
        request_parts_dir.mkdir(parents=True, exist_ok=True)
        
        # 4. [핵심] 마법 주머니에 '이 요청만을 위한 경로' 넣기
        _output_dir_ctx.set(request_output_dir)
        _parts_dir_ctx.set(request_parts_dir)
        
        print(f"[Config] Request Path Initialized: {request_output_dir}")

    # --- [추가 4] 모델들이 가져다 쓸 '가짜 변수(Property)' ---
    @property
    def OUTPUT_DIR(self) -> Path:
        """
        모델이 config.OUTPUT_DIR 을 부르면, 
        자동으로 현재 요청의 ContextVar에서 값을 꺼내줌
        """
        return _output_dir_ctx.get()

    @property
    def DIR_PARTS(self) -> Path:
        return _parts_dir_ctx.get()