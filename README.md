# 백령도 해안 경계 AI 시스템 - 실행 가이드

## 1. 환경 설정

### 필수 패키지 설치
```bash
pip install ultralytics
pip install fastapi uvicorn
pip install streamlit
pip install openai
pip install python-dotenv
pip install opencv-python
pip install pillow
pip install pandas
pip install pyyaml
pip install tqdm
```

### 환경 변수 설정
프로젝트 루트에 `.env` 파일 생성:
```
OPENAI_API_KEY=your-api-key-here
```

---

## 2. 데이터 전처리

### 실행
```bash
cd C:\Army_project
python data_tools/json2Yolo.py
```

### 결과
- Train/Val JSON 라벨을 YOLO txt로 변환
- 불량 데이터 필터링
- `data_filtered.yaml` 파일 생성
- 최종 데이터셋: `data/Filtered/` 폴더에 저장

### 소요 시간
약 8시간 (8 워커 기준)

---

## 3. 모델 학습

### YOLOv11 학습
```bash
yolo train model=yolo11s.pt data=C:/Army_project/data/data_filtered.yaml epochs=100 imgsz=640 batch=16
```

### 학습 완료 후
`runs/detect/train/weights/best.pt` 파일 생성됨

### best.pt 파일 이동
```bash
# 생성된 모델을 프로젝트 폴더로 이동
mkdir runs/army_project_clean_yolo11s/weights
copy runs/detect/train/weights/best.pt runs/army_project_clean_yolo11s/weights/
```

---

## 4. 서버 실행

### FastAPI 서버 실행 (터미널 1)
```bash
cd C:\Army_project
uvicorn main_api:app --host 0.0.0.0 --port 8000 --reload
```

### Streamlit UI 실행 (터미널 2)
```bash
cd C:\Army_project
streamlit run app.py
```

### 접속 확인
- FastAPI 문서: http://localhost:8000/docs
- Streamlit UI: http://localhost:8501
- Health Check: http://localhost:8000/health

---

## 5. 로그 및 리포트

### 탐지 로그 실시간 확인
```bash
tail -f detections.log
```

### 월간 리포트 생성
```bash
python generate_monthly_report.py
```

---

## 트러블슈팅

### 문제 1: OpenAI API 키 오류
**증상**: Invalid API key

**해결**:
```bash
# .env 파일 확인
type .env
```

### 문제 2: 모델 파일 없음
**증상**: FileNotFoundError: best.pt

**해결**:
1. 모델 학습 완료 여부 확인
2. 경로 확인: `runs/army_project_clean_yolo11s/weights/best.pt`
3. 백업 모델 자동 로드됨 (yolov8n.pt)

### 문제 3: CUDA 메모리 부족
**증상**: CUDA out of memory

**해결**:
```bash
# batch size 줄이기
yolo train ... batch=8

# 또는 CPU 사용
yolo train ... device=cpu
```

### 문제 4: 포트 충돌
**증상**: Address already in use

**해결**:
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID [PID번호] /F

# 다른 포트 사용
uvicorn main_api:app --port 8001
```

---

## 파일 구조

```
C:/Army_project/
├── data/
│   ├── Train/
│   │   ├── json/          # 원본 JSON 라벨
│   │   ├── Origin/        # 원본 이미지
│   │   └── labels/        # 변환된 YOLO txt
│   ├── Val/
│   │   ├── json/
│   │   ├── Origin/
│   │   └── labels/
│   ├── Filtered/          # 필터링된 최종 데이터
│   │   ├── Train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── Val/
│   │       ├── images/
│   │       └── labels/
│   └── data_filtered.yaml # YOLO 학습 설정
├── data_tools/
│   └── json2Yolo.py       # 전처리 스크립트
├── modules/
│   ├── llm_module.py      # LLM + TTS
│   └── main.py            # FastAPI 메인
├── runs/
│   └── army_project_clean_yolo11s/
│       └── weights/
│           └── best.pt    # 학습된 모델
├── main_api.py            # FastAPI 서버
├── app.py            # Streamlit UI
├── generate_monthly_report.py  # 리포트 생성
├── detections.log         # 탐지 로그
└── .env                   # API 키
```

---

## 빠른 시작

```bash
# 1. 패키지 설치
pip install ultralytics fastapi uvicorn streamlit openai python-dotenv opencv-python pillow pandas pyyaml tqdm

# 2. API 키 설정
echo OPENAI_API_KEY=your-key > .env

# 3. 데이터 전처리 (최초 1회)
python data_tools/json2Yolo.py

# 4. 모델 학습 (최초 1회)
yolo train model=yolo11s.pt data=C:/Army_project/data/data_filtered.yaml epochs=100

# 5. 서버 실행 (터미널 1)
uvicorn main_api:app --host 0.0.0.0 --port 8000

# 6. UI 실행 (터미널 2)
streamlit run app.py
```

---

## 주의사항

- 데이터 전처리는 최초 1회만 실행
- 모델 학습은 GPU 권장 (CPU는 매우 느림)
- OpenAI API 사용 시 과금 주의
- detections.log 파일 주기적 백업 권장
