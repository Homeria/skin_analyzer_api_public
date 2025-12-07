from __future__ import annotations
import os, warnings, re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import json
import mediapipe as mp
from app.core.config import Config


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
    absl_logging.use_python_logging()
except Exception:
    pass
warnings.filterwarnings("ignore", category=UserWarning)


FULL_H, FULL_W = 1440, 1080   
FACE_H, FACE_W = 960,  720   

PART_ID_TO_STATS = {
    1: "forehead",
    2: "glabella",
    3: "left_eye",
    4: "right_eye",
    5: "left_cheek",
    6: "right_cheek",
    7: "nose_mouth",
    8: "chin",
}
STATS_TO_LABEL = {
    "forehead":"forehead",
    "glabella":"glabella",
    "left_eye":"left_eye",
    "right_eye":"right_eye",
    "left_cheek":"left_cheek",
    "right_cheek":"right_cheek",
    "nose_mouth":"nose_mouth",
    "chin":"chin",
}
KOR_PART = {
    "forehead":"이마", "glabella":"미간",
    "left_eye":"왼쪽 눈가", "right_eye":"오른쪽 눈가",
    "left_cheek":"왼쪽 볼", "right_cheek":"오른쪽 볼",
    "nose_mouth":"코/입", "chin":"턱",
}

PART_PREFIXES = {
    "forehead":   ("forehead_",),
    "glabella":   ("glabella_","glabellus_"),
    "left_eye":   ("left_eye_","l_eye_","l_perocular_","left_periocular_","left_periorbital_"),
    "right_eye":  ("right_eye_","r_eye_","r_perocular_","right_periocular_","right_periorbital_"),
    "left_cheek": ("left_cheek_","l_cheek_"),
    "right_cheek":("right_cheek_","r_cheek_"),
    "nose_mouth": ("nose_mouth_","lip_","lips_","mouth_nose_","nosemouth_"),
    "chin":       ("chin_",),
}

LOWER_IS_BETTER = {"wrinkle","pore","pore_count","pigment","dryness","sagging"}

def decode_image_bytes(data_bytes: bytes) -> np.ndarray:
    # 1. 바이트 스트림을 1차원 uint8 배열로 변환 (기존 np.fromfile과 같은 역할)
    nparr = np.frombuffer(data_bytes, np.uint8)
    
    # 2. 이미지로 디코딩 (기존 로직 동일)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 3. 디코딩 실패 시 (이미지 파일이 아님)
    if img is None:
        raise ValueError("이미지 디코딩 실패: 유효하지 않은 이미지 데이터입니다.")
        
    return img

def imread_unicode(p: Path) -> np.ndarray:
        data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return img
    
def imwrite_unicode(p: Path, img: np.ndarray) -> bool:
    ext = p.suffix if p.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(p))
    return True

def mosaic_eyes_with_mediapipe_bgr(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[Tuple[int,int,int,int]]]:
    h,w = img_bgr.shape[:2]
    # 눈 가림용 MediaPipe 초기화
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        res = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return img_bgr, None
        lm = res.multi_face_landmarks[0].landmark
        left_ids  = [33,133,160,159,158,144,145,153,154,155]
        right_ids = [362,263,387,386,385,373,374,380,381,382]
        def box(ids):
            xs = [lm[i].x*w for i in ids]; ys = [lm[i].y*h for i in ids]
            return min(xs), min(ys), max(xs), max(ys)
        lx1,ly1,lx2,ly2 = box(left_ids)
        rx1,ry1,rx2,ry2 = box(right_ids)
        x1 = min(lx1, rx1); y1 = min(ly1, ry1)
        x2 = max(lx2, rx2); y2 = max(ly2, ry2)
        bx1,by1,bx2,by2 = clamp_box(x1,y1,x2-x1,y2-y1, w,h)
        cv2.rectangle(img_bgr, (bx1,by1), (bx2,by2), (0,0,0), -1)
        return img_bgr, (bx1,by1,bx2,by2)

def resize_to_full(img_bgr: np.ndarray) -> Tuple[np.ndarray,float,int,int]:
    h,w = img_bgr.shape[:2]
    scale = min(FULL_W/w, FULL_H/h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    pad_w, pad_h = FULL_W-nw, FULL_H-nh
    l, r = pad_w//2, pad_w - pad_w//2
    t, b = pad_h//2, pad_h - pad_h//2
    out = cv2.copyMakeBorder(resized, t,b,l,r, cv2.BORDER_CONSTANT, value=(0,0,0))
    return out, scale, l, t

def clamp_box(x, y, w, h, img_w, img_h):
    x1 = max(0, min(img_w-1, int(x)))
    y1 = max(0, min(img_h-1, int(y)))
    x2 = max(0, min(img_w-1, int(x+w)))
    y2 = max(0, min(img_h-1, int(y+h)))
    if x2 <= x1: x2 = min(img_w-1, x1+1)
    if y2 <= y1: y2 = min(img_h-1, y1+1)
    return x1,y1,x2,y2

def bgr_to_tensor(img_bgr: np.ndarray, device) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    x = torch.from_numpy(np.transpose(rgb,(2,0,1))).unsqueeze(0).to(device)
    return x

def bgr_to_norm_tensor(img_bgr: np.ndarray, device) -> torch.Tensor:
    x = bgr_to_tensor(img_bgr, device)
    mean = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
    return (x-mean)/std

def resize_face_to_inner(img_bgr: np.ndarray) -> Tuple[np.ndarray,float,int,int]:
    h,w = img_bgr.shape[:2]
    scale = min(FACE_W/w, FACE_H/h)
    nw, nh = int(round(w*scale)), int(round(h*scale))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_CUBIC)
    pad_w, pad_h = FACE_W-nw, FACE_H-nh
    l, r = pad_w//2, pad_w - pad_w//2
    t, b = pad_h//2, pad_h - pad_h//2
    out = cv2.copyMakeBorder(resized, t,b,l,r, cv2.BORDER_CONSTANT, value=(0,0,0))
    return out, scale, l, t

def decode_gender_age_from_demo(demo_out) -> Tuple[str,int]:
    if isinstance(demo_out, dict):
        g_logits = demo_out["gender"]
        a_pred   = demo_out["age"]
    else:
        g_logits, a_pred = demo_out
    g_idx = int(torch.argmax(g_logits, dim=1).item())
    gender = "M" if g_idx == 0 else "F"
    age = int(round(float(a_pred.squeeze().item())))
    age = max(10, min(90, age))
    return gender, age

def crop_square_224(img: np.ndarray, box_xywh: List[float], margin: float=0.12) -> np.ndarray:
    H,W = img.shape[:2]
    x,y,w,h = box_xywh
    cx, cy = x+w/2, y+h/2
    s = max(w,h)*(1.0+margin*2)
    x1, y1 = int(round(cx-s/2)), int(round(cy-s/2))
    x2, y2 = int(round(cx+s/2)), int(round(cy+s/2))
    x1,y1,x2,y2 = clamp_box(x1,y1,x2-x1,y2-y1, W,H)
    patch = img[y1:y2, x1:x2].copy()
    if patch.size == 0:
        return np.zeros((224,224,3), np.uint8)
    return cv2.resize(patch, (224,224), interpolation=cv2.INTER_AREA)

def is_allowed_for_part(stats_name: str, key: str, is_cls: bool) -> bool:
    prefs = PART_PREFIXES.get(stats_name, ())
    k = key.lower()
    if not any(k.startswith(p) for p in prefs):
        return False
    return True

def canonical_metric_for_pkl(stats_name: str, key_raw: str) -> str:
    k = key_raw.lower()
    for pref in PART_PREFIXES.get(stats_name, ()):
        if k.startswith(pref):
            k = k[len(pref):]
            break
    if k.startswith(stats_name + "_"):
        k = k[len(stats_name)+1:]
    k = k.replace("rz=rtm", "rz")
    if k.startswith("wrinkle_"):
        sub = k.split("wrinkle_")[-1].upper()
        return f"wrinkle_{sub}"
    k = re.sub(r"elasticity_q(\d+)", lambda m: f"elasticity_Q{m.group(1)}", k)
    k = re.sub(r"elasticity_r(\d+)", lambda m: f"elasticity_R{m.group(1)}", k)
    if k in ("pore","pore_count"):
        return "pore_count"
    if k == "moisture":
        return "moisture"
    return k

def black_ratio(img_bgr: np.ndarray) -> float:
    nz = np.count_nonzero(np.any(img_bgr>0, axis=2))
    total = img_bgr.shape[0]*img_bgr.shape[1]
    return 1.0 - (nz/float(total) if total>0 else 1.0)

def pretty_metric(k: str) -> str:
    lk = k.lower()
    mapping = {
        "wrinkle_grade":"주름 등급","pigmentation_grade":"색소 등급","pore_grade":"모공 등급",
        "dryness_grade":"건조도 등급","sagging_grade":"처짐 등급",
        "moisture":"수분","pore_count":"모공 수"
    }
    if "elasticity_q" in lk:
        n = lk.split("elasticity_q")[-1]
        return f"탄력(Q{n})"
    if "elasticity_r" in lk:
        n = lk.split("elasticity_r")[-1]
        return f"탄력(R{n})"
    if "wrinkle_" in lk and "grade" not in lk:
        tail = k.split("_")[-1].upper()
        return f"주름({tail})"
    return mapping.get(lk, k)

def render_and_save(img_stem: str,
                    gender: str, age: int,
                    result: Dict[str,Any],
                    OUT_TXT,
                    OUT_JSON,
                    INPUT_DIR):
    lines = [f"[DEMO] 성별: {gender} / 나이: {age}세",""]
    for r in result["regions"]:
        kor = KOR_PART.get(r["part_name"], r["part_name"])
        lines.append(f"[{kor}]")
        for k,v in (r.get("regression") or {}).items():
            pretty = pretty_metric(canonical_metric_for_pkl(r['stats_name'], k))
            lines.append(f" - {pretty}: {v:.4f}")
        for k,g in (r.get("classification") or {}).items():
            pretty = pretty_metric(canonical_metric_for_pkl(r['stats_name'], k))
            lines.append(f" - {pretty}: 등급 {int(g)}")
        lines.append("")
    (OUT_TXT / f"{img_stem}_diag.txt").write_text("\n".join(lines).strip(), encoding="utf-8")

    ex = [f"[DEMO] 성별: {gender} / 나이: {age}세"]
    for r in result["regions"]:
        kor = KOR_PART.get(r["part_name"], r["part_name"])
        ex.append(f"\n[{kor}]")
        reg = r.get("regression") or {}
        pct = r.get("percentile") or {}
        grd = r.get("grade") or {}
        for k,v in reg.items():
            m = canonical_metric_for_pkl(r['stats_name'], k)
            pretty = pretty_metric(m)
            p = pct.get(m, float("nan"))
            g = grd.get(m, "N/A")
            ex.append(f" - {pretty}: {v:.4f} → 백분위 {p:.1f}% / 등급 {g}")
        for k,g in (r.get("classification") or {}).items():
            pretty = pretty_metric(canonical_metric_for_pkl(r['stats_name'], k))
            ex.append(f" - {pretty}: 등급 {int(g)}")
        if r.get("patch_black_ratio",0.0) > 0.15:
            ex.append("   (참고: 패치 검정영역 비율이 높아 측정 신뢰도가 낮을 수 있습니다.)")
    (OUT_TXT / f"{img_stem}_explain.txt").write_text("\n".join(ex).strip(), encoding="utf-8")

    parts_json=[]
    for r in result["regions"]:
        parts_json.append({
            "id": r["part_index"],
            "name": r["part_name"],
            "bbox": r["bbox_xywh_face"],
            "attrs": {
                "equipment": r.get("regression", {}),
                "percentile": r.get("percentile", {}),
                "grade": r.get("grade", {}),
                "annotations": r.get("classification", {}),
                "patch_black_ratio": r.get("patch_black_ratio", 0.0)
            }
        })
    with open(OUT_JSON / f"{img_stem}.json", "w", encoding="utf-8") as f:
        json.dump({
            "image": str((INPUT_DIR / img_stem).with_suffix("")),
            "info": {"gender": gender, "age": age},
            "parts": parts_json
        }, f, ensure_ascii=False, indent=2)

def make_dirs(config: Config):
    dirs_to_create = [
            config.INPUT_DIR,         # 입력 이미지 저장 경로 (root/request/timestamp)
            config.OUTPUT,            # 결과 저장 루트 (root/response/timestamp)
            config.OUT_VIS,           # 결과물 하위 폴더들...
            config.OUT_TXT,
            config.OUT_JSON,
            config.OUT_FSDD,
            config.OUT_COMBINED_DIR
        ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
            
    print(f"[Config] Created directories for timestamp: {config.TIMESTAMP}")

def draw_face_boxes(img, boxes, save_path):
    pass

def save_debug_image(
    path: Path, 
    img_bgr: np.ndarray, 
    boxes_xywh: List[List[float]], 
    labels: Optional[List[str]] = None,
    color: Tuple[int, int, int] = (0, 255, 255)
) -> None:
    """
    이미지에 박스와 라벨을 그려서 저장하는 디버깅용 함수
    - path: 저장할 파일 경로 (Path 객체)
    - img_bgr: 원본 이미지 (cv2 BGR)
    - boxes_xywh: [x, y, w, h] 형태의 좌표 리스트
    - labels: 각 박스에 달릴 텍스트 리스트 (없으면 생략)
    """
    if img_bgr is None: 
        return

    vis = img_bgr.copy()
    img_h, img_w = vis.shape[:2]

    for i, box in enumerate(boxes_xywh):
        x, y, w, h = box
        # 이미지 밖으로 나가지 않게 좌표 보정 (기존 clamp_box 활용)
        x1, y1, x2, y2 = clamp_box(x, y, w, h, img_w, img_h)

        # 1. 박스 그리기
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # 2. 텍스트(라벨) 그리기
        if labels and i < len(labels):
            text = labels[i]
            # 텍스트가 잘 보이게 박스 위쪽에 배치 (공간 없으면 안쪽으로)
            txt_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            cv2.putText(vis, text, (x1, txt_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    # 저장 (기존 imwrite_unicode 활용)
    imwrite_unicode(path, vis)

def json_default(obj):
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, 'isoformat'): # datetime 등
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def safe_crop(
    img_bgr: np.ndarray, 
    box_xywh: List[float]
) -> np.ndarray:
    """
    이미지 영역 밖으로 나가지 않게 안전하게 크롭하는 함수
    - box_xywh: [x, y, w, h]
    """
    h, w = img_bgr.shape[:2]
    x, y, bw, bh = box_xywh
    
    x1, y1, x2, y2 = clamp_box(x, y, bw, bh, w, h)
    
    crop = img_bgr[y1:y2, x1:x2].copy()
    
    # 만약 크롭 영역이 비어있으면(0size) 검은색 1x1 픽셀 반환 (에러 방지)
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
        
    return crop