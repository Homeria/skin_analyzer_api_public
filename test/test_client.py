import requests
import os
import json

# --- 설정 ---
SERVER_URL = "http://127.0.0.1:8000/analyze/skin"
IMAGE_PATH = "test_face.jpg"

def run_test():
    # 1. 이미지 파일 존재 확인
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ 에러: '{IMAGE_PATH}' 파일을 찾을 수 없습니다.")
        print("테스트할 얼굴 사진을 스크립트와 같은 폴더에 저장해주세요.")
        return

    # 2. 함께 보낼 사용자 정보 (Form Data)
    form_data = {
        "gender": "F",      # 프론트엔드에서 보낼 예상 값
        "birth_year": 1995,
        "birth_month": 10,
        # 리스트는 requests가 알아서 'concerns=주름&concerns=건조' 형태로 변환해줍니다.
        "concerns": ["주름", "건조", "모공"] 
    }

    # 3. 이미지 파일 준비
    with open(IMAGE_PATH, "rb") as image_file:
        files = {
            "image": (os.path.basename(IMAGE_PATH), image_file, "image/jpeg")
        }
        
        print(f"🚀 '{IMAGE_PATH}' 전송 중... (서버가 AI 모델을 돌리느라 시간이 좀 걸릴 수 있습니다)")
        
        try:
            # 4. POST 요청 전송 (API Key 헤더 제거됨)
            response = requests.post(SERVER_URL, files=files, data=form_data)
            
            # 상태 코드 확인
            response.raise_for_status()
            
            # 응답 JSON 파싱
            result = response.json()

            print("\n✅ 요청 성공!")
            print("="*40)
            
            # --- [1] 인구통계 정보 출력 ---
            demo = result.get("demo", {})
            print(f"👤 [인구통계 예측]")
            print(f"   - 성별: {demo.get('gender')}")
            print(f"   - 나이: {demo.get('age')}세")
            print("-" * 40)

            # --- [2] 추천 문구 출력 ---
            print(f"🧴 [추천 스킨케어]")
            for rec in result.get("recommendations", []):
                print(f"   - {rec}")
            print("-" * 40)

            # --- [3] 상세 진단 결과 출력 (양이 많으니 요약해서 출력) ---
            diag = result.get("diag", {})
            regions = diag.get("regions", [])
            print(f"📊 [상세 진단 결과] (총 {len(regions)}개 부위 분석됨)")
            
            # 예시로 첫 번째 부위(보통 이마)의 데이터만 상세히 출력
            if regions:
                first_part = regions[0]
                p_name = first_part.get('part_name')
                print(f"\n   📍 예시: '{p_name}' 부위 상세 데이터")
                
                # 측정값(Regression) 일부 출력
                print(f"      [측정값 (Raw)]")
                regs = first_part.get('regression', {})
                for k, v in list(regs.items())[:3]: # 3개만 예시로 출력
                    print(f"        - {k}: {v:.4f}")
                
                # 백분위(Percentile) 일부 출력
                print(f"      [백분위 (상위 %)]")
                pcts = first_part.get('percentile', {})
                for k, v in list(pcts.items())[:3]:
                    print(f"        - {k}: 상위 {v:.1f}%")

            print("\n(나머지 부위 데이터는 생략함)")
            print("="*40)

        except requests.exceptions.HTTPError as e:
            print(f"\n❌ HTTP 에러 발생: {e}")
            print(f"   응답 내용: {e.response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"\n❌ 서버 연결 실패: {e}")

if __name__ == "__main__":
    run_test()