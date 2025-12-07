
import torch
from pathlib import Path
from datetime import datetime

class Config2:

    def __init__(self):

        self.ROOT = Path(__file__).resolve().parent.parent.parent

        self.TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.DATA_DIR = self.ROOT / "data"
        self.OUTPUT_DIR = self.DATA_DIR / self.TIMESTAMP
        self.DIR_PARTS = self.OUTPUT_DIR / "parts"

        self.APP = self.ROOT / "app"

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

    def update_time(self):

        self.TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.OUTPUT_DIR = self.DATA_DIR / self.TIMESTAMP
        self.DIR_PARTS = self.OUTPUT_DIR / "parts"
        

    def _make_dirs(self):

        """
        저장에 필요한 폴더들을 미리 생성
        """

        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DIR_PARTS.mkdir(parents=True, exists_ok=True)
        print(f"[Config] Output Directory Created: {self.OUTPUT_DIR}")



