# 피부 분석 AI API

이 프로젝트는 얼굴 이미지를 분석하여 성별, 피부 나이, 피부 컨디션, 피부 질환을 추론하는 AI API 서버입니다.

## ⚙️ 개발 환경 설정

1. **파이썬 버전**
    Python 3.10.11(Windows) - https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe

2.  **가상환경 생성**
    ```bash
    python -m venv venv

3. **가상환경 활성화**
    ```bash
    source venv/bin/activate  # macOS/Linux
    .\venv\Scripts\activate   # Windows
    ```

4.  **필요한 라이브러리 설치 (가상환경 내 1회 최초 실행)**
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ 서버 실행 방법
    # 가상환경 활성화
    .\venv\Scripts\activate 

    # 1. localhost로 열 때 - 8000번 포트가 기본 -> localhost:8000
    uvicorn app.main:app --reload 

    # 2. 외부망으로 열 때 - port_number 지정 가능 -> IP:port_number
    uvicorn app.main:app --host 0.0.0.0 --port port_number --reload

## 체크포인트 다운로드 및 배치
    체크포인트들은 용량으로 인해 깃허브에 올리지 못하므로, 다음의 링크에서 다운로드 받으세요.
    https://drive.google.com/file/d/1xIhWo2YqqSvRAS_9u9qQpuYwMzCbXWdm/view?usp=sharing
    체크포인트를 받은 후 해당 Checkpoints 폴더 자체를 app/models/에 붙여넣으세요.
    예시 : 얼굴 탐지 체크포인트의 경로는 skin_analyzer_api_public/app/models/checkpoints/FACE_CKPT.pth

## 추론 Post 요청 시 저장되는 경로
    /data/[Timestamp]/
    예시 : 2025년 11월 27일 02시 35분 27초 경 POST -> skin_analyzer_api_public/data/20251127_023527/

## API 사용 설명서
    해당 API 도메인에 /docs 또는 /redoc을 통하여 접근합니다.
    예시 : 서버 실행 방법 1번을 택했을 때 -> localhost:8000/docs 또는 localhost:8000/redoc
    docs : [Try it out] 기능을 통해 요청 테스트가 가능합니다.
    redoc : 상대적으로 요청에 대한 문서가 docs보다 잘 정리되어 있습니다.

### mediapipe DDL 관련 이슈 해결 방법
해당 이슈는 Microsoft Visual C++ 재배포 가능 패키지가 설치되지 않아 생기므로, 해결 방법은 2가지입니다.
1. Visual C++ 재배포 가능 패키지 다운로드
https://learn.microsoft.com/ko-kr/cpp/windows/latest-supported-vc-redist?view=msvc-170
2. 가상환경에서 해당 명령어를 통해 파이썬 패키지로 강제 주입
    ```bash
    pip install msvc-runtime
    ```
