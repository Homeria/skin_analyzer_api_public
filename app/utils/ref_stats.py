import pickle
import math
import bisect
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from app.core.config import Config
from app.utils.image_utils import LOWER_IS_BETTER

class RefStats:

    """
    통계 데이터(Pickle)를 로드하여, 측정값에 대한 백분위(Percentile)와 등급(Grade)을 산출하는 클래스
    """
    
    REGION_LOCKED = {"moisture", "pore_count"}

    def __init__(self, config: Config):
        self.data = {}
        pkl_path = config.REF_PKL

        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                self.data = pickle.load(f)
        else:
            print(f"[Warning] Reference Stats Pickle not found at: {pkl_path}")

        # 데이터 정렬 전처리 (Lookup 속도 향상)
        for k, d in list(self.data.items()):
            if isinstance(d, dict) and "values" in d and not d.get("_sorted", False):
                try:
                    d["values"] = sorted(d["values"])
                    d["_sorted"] = True
                except Exception:
                    pass

    @staticmethod
    def _percentile_hazen(sorted_arr, value: float) -> float:
        """Hazen Formula: p = 100 * (rank - 0.5) / n"""
        if not sorted_arr:
            return float("nan")
        i = bisect.bisect_right(sorted_arr, value)
        n = len(sorted_arr)
        p = 100.0 * (i - 0.5) / n
        return max(0.0, min(100.0, p))

    @staticmethod
    def _age_bucket_soft(age: Optional[int]) -> List[Tuple[str, str, float]]:
        """나이대(Decade)와 구간(Early/Mid/Late)을 부드럽게 보간"""
        if not isinstance(age, int):
            return [("any", "any", 1.0)]
        
        decade_val = (age // 10) * 10
        if decade_val < 10: decade = "10s"
        elif decade_val > 80: decade = "80s"
        else: decade = f"{decade_val}s"

        u = age % 10
        stage = "early" if u <= 2 else ("mid" if u <= 6 else "late")
        base = (decade, stage, 0.6 if (u >= 7 or u <= 2) else 1.0)
        
        neighbors = []
        # (기존 복잡한 이웃 계산 로직 유지 - 생략 없이 원본 로직 사용 권장)
        # 간소화를 위해 핵심 로직만 유지하거나 원본 그대로 복사 사용
        if u >= 7:
            # 다음 연대 Early와 섞음
            next_dec = decade_val + 10
            if next_dec <= 80:
                neighbors.append((f"{next_dec}s", "early", 0.4))
        elif u <= 2 and decade_val >= 20:
            # 이전 연대 Late와 섞음
            prev_dec = decade_val - 10
            neighbors.append((f"{prev_dec}s", "late", 0.4))

        if neighbors and base[2] < 1.0:
            return [base] + neighbors
        return [(decade, stage, 1.0)]

    def _region_candidates(self, region: str, metric: str) -> List[str]:
        if metric in self.REGION_LOCKED:
            return [region, "any"]
        return [region, "face", "any"]

    def lookup(self, sex: Optional[str], age: Optional[int], region: str, metric: str,
               value: float,
               reg_mean: Dict[str, float], reg_std: Dict[str, float],
               fallback_key: Optional[str] = None) -> Dict[str, Any]:
        """
        핵심 조회 메서드
        Args:
            value: 모델이 예측한 Raw Value (Denormalized 된 값)
            reg_mean, reg_std: 모델 메타데이터 (Z-score 백업용)
        """
        sex_b = (sex or "any").upper()
        age_buckets = self._age_bucket_soft(age)
        regions = self._region_candidates(region, metric)

        found = False
        acc_p = 0.0
        acc_w = 0.0

        # 1. DB(Pickle)에서 통계 분포 조회
        for decade, stage, w in age_buckets:
            for rg in regions:
                key = (sex_b, decade, stage, rg, metric)
                d = self.data.get(key)
                if not (d and d.get("values")):
                    continue
                
                vals = d["values"]
                p = self._percentile_hazen(vals, float(value))
                
                # '좋은 값'의 방향성 체크
                better = d.get("better", None)
                if better in (-1, "lower") or (better is None and any(t in metric.lower() for t in LOWER_IS_BETTER)):
                    p = 100.0 - p
                
                acc_p += p * w
                acc_w += w
                found = True
                break

        # 2. DB에 데이터가 없으면 정규분포(Z-score) 근사 계산
        if not found:
            mu = reg_mean.get(metric) or (reg_mean.get(fallback_key, 0.0) if fallback_key else 0.0)
            sd = reg_std.get(metric) or (reg_std.get(fallback_key, 1.0) if fallback_key else 1.0)
            if sd <= 1e-6: sd = 1.0
            
            z = (float(value) - float(mu)) / float(sd)
            z = max(-6.0, min(6.0, z)) # Clamp
            p = 50.0 * (1.0 + math.erf(z / math.sqrt(2.0)))
            
            if any(t in metric.lower() for t in LOWER_IS_BETTER):
                p = 100.0 - p
            acc_p, acc_w = p, 1.0

        pct = float(max(0.0, min(100.0, acc_p / acc_w if acc_w > 0 else acc_p)))

        # 3. 등급 산출
        if pct >= 95: grade = "S+"
        elif pct >= 90: grade = "S0"
        elif pct >= 80: grade = "A+"
        elif pct >= 70: grade = "A0"
        elif pct >= 60: grade = "B+"
        elif pct >= 50: grade = "B0"
        elif pct >= 35: grade = "C+"
        elif pct >= 20: grade = "C0"
        else: grade = "D0"

        return {"percentile": pct, "grade": grade}