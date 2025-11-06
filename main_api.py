from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
# TTS 기능이 포함된 LLM 모듈을 import합니다.
from modules.llm_module import generate_warning, format_warning_text 
import cv2
import numpy as np
import logging
from typing import Dict, List
from datetime import datetime
import json # JSON 로깅을 위해 추가

# --- 기본 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- 탐지 기록용 로거 설정 ('detections.log' 파일) ---
detection_logger = logging.getLogger('detection_logger')
detection_logger.setLevel(logging.INFO)
# 서버 리로드 시 핸들러가 중복 추가되는 것을 방지
if not detection_logger.handlers:
    f_handler = logging.FileHandler('detections.log', encoding='utf-8')
    f_format = logging.Formatter('%(message)s') # 로그 파일에는 JSON 내용만 기록
    f_handler.setFormatter(f_format)
    detection_logger.addHandler(f_handler)

app = FastAPI(
    title="백령도 해안 경계 AI 시스템 (TTS/로깅 포함)",
    description="드론 영상 기반 객체 탐지 및 LLM 경고 생성 API (TTS 음성, 탐지 결과 로깅)",
    version="1.2.0"
)

# --- CORS 설정 ---
# Streamlit UI (보통 8501 포트) 등 다른 도메인에서의 요청을 허용합니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 배포 시에는 특정 도메인(Streamlit 주소)으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 전역 변수 설정 ---
yolo = None
CLASS_NAMES = { 
    0: "어선", 
    1: "상선", 
    2: "군함", 
    3: "사람", 
    4: "유조류" 
}
DISTANCE_THRESHOLDS = { 
    "critical": 0.5,   # '매우 가까움' (이미지 높이의 50% 초과)
    "warning": 0.2     # '중간 거리' (이미지 높이의 20% 초과)
}

# --- 서버 시작 시 모델 로드 ---
@app.on_event("startup")
def load_model():
    """서버가 시작될 때 YOLO 모델을 메모리에 미리 로드합니다."""
    global yolo
    try:
        model_path = "runs/army_project_clean_yolo11s/weights/best.pt"
        yolo = YOLO(model_path)
        logging.info(f"YOLO 커스텀 모델 로드 성공: {model_path}")
    except Exception as e:
        # 커스텀 모델 로드 실패 시, 백업용 기본 모델 로드
        logging.warning(f"커스텀 모델 로드 실패: {e} | 백업 모델(yolov8n.pt) 사용")
        yolo = YOLO("yolov8n.pt")
        logging.info("백업 모델(yolov8n.pt) 로드 완료")


def calculate_distance_status(box_height: float, img_height: float) -> str:
    """
    객체의 높이 비율로 상대적 거리 판정 (도메인 지식 기반)
    Z값(깊이)이 없는 2D 이미지를 위한 현실적 대안입니다.
    """
    ratio = box_height / img_height
    
    if ratio > DISTANCE_THRESHOLDS["critical"]:
        return "매우 가까움"
    elif ratio > DISTANCE_THRESHOLDS["warning"]:
        return "중간 거리"
    else:
        return "멀리 있음"


def process_yolo_results(results) -> List[Dict]:
    """YOLO 추론 결과를 표준화된 JSON 리스트로 변환"""
    detections = []
    
    if not results or results[0].boxes is None:
        return detections
    
    # 원본 이미지 크기
    img_h, img_w = results[0].orig_shape
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
        
        # 바운딩 박스 높이
        box_height = xyxy[3] - xyxy[1]
        
        # 거리 판정
        distance_status = calculate_distance_status(box_height, img_h)
        
        detection = {
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
            "confidence": round(confidence, 2),
            "bbox": [round(x, 1) for x in xyxy],
            "box_size": { 
                "width": round(xyxy[2] - xyxy[0], 1), 
                "height": round(box_height, 1) 
            },
            "distance_status": distance_status
        }
        detections.append(detection)
    
    return detections


# --- 메인 API 엔드포인트 ---
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    이미지를 받아 객체 탐지(YOLO), 전술 경고(LLM), 음성(TTS)을 생성하고
    탐지 결과를 로깅합니다.
    """
    start_time = datetime.now()
    
    # 1. 이미지 검증
    if not file.content_type.startswith("image/"):
        logging.warning(f"잘못된 파일 타입 업로드: {file.content_type}")
        raise HTTPException(
            status_code=400, 
            detail=f"이미지 파일만 업로드 가능합니다. (현재: {file.content_type})"
        )
    
    # 2. 이미지 디코딩
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None: 
            raise ValueError("이미지 디코딩 실패 (cv2.imdecode 반환 값 None)")
        logging.info(f"이미지 로드 성공: {file.filename} | 크기: {img.shape}")
    except Exception as e:
        logging.error(f"이미지 처리 오류: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"이미지 파일을 처리할 수 없습니다: {str(e)}"
        )
    
    # 3. YOLO 추론
    try:
        results = yolo.predict(img, verbose=False, conf=0.25)
        detections = process_yolo_results(results)
        logging.info(f"탐지 완료: {len(detections)}개 객체")
    except Exception as e:
        logging.error(f"YOLO 추론 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"객체 탐지 중 서버 오류 발생: {str(e)}"
        )

    # 4. LLM + TTS 경고 생성
    # LLM에 전달할 탐지 객체 리스트 생성
    detected_objects = [f"{d['class_name']} → {d['distance_status']}" for d in detections]
    
    try:
        # llm_module_with_tts.py의 함수 호출
        # 이 warning 딕셔너리 안에 audio_base64가 포함되어 있음
        warning = generate_warning(detected_objects) 
        logging.info(f"경고 생성(TTS포함) 완료: [{warning.get('level', 'N/A')}]")
    except Exception as e:
        # LLM/TTS 호출 실패 시에도 서비스는 중단되지 않음 (폴백)
        logging.error(f"LLM(TTS) 경고 생성 오류: {e}")
        warning = { 
            "level": "오류", 
            "summary": "경고 메시지 생성 실패", 
            "action": "수동 확인 필요", 
            "source": "error", 
            "audio_base64": None # 오디오 없음
        }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # 5. 최종 응답 데이터 생성
    response_data = {
        "status": "success",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "processing_time": round(elapsed, 2),
        "image_info": { 
            "filename": file.filename, 
            "size": img.shape[:2] # [height, width]
        },
        "detections": detections,
        "detected_objects": detected_objects,
        "warning": warning # 'audio_base64'가 포함된 경고 딕셔너리
    }

    # 6. 통계용 로그 기록
    # '주의' 또는 '경보' 레벨일 때만 'detections.log' 파일에 기록
    log_level = warning.get("level", "안전")
    if log_level in ["경보", "주의"]:
        try:
            # 로그에 남길 데이터만 간추림 (개인정보, 불필요한 데이터 제외)
            log_data = {
                "timestamp": response_data["timestamp"],
                "level": log_level,
                "summary": warning.get("summary", "N/A"),
                "action": warning.get("action", "N/A"),
                "detected_objects": detected_objects,
                "filename": file.filename
            }
            # JSON 문자열로 변환하여 로그 파일에 씀
            detection_logger.info(json.dumps(log_data, ensure_ascii=False))
        except Exception as e:
            logging.error(f"탐지 로그 파일 쓰기 오류: {e}")

    return JSONResponse(content=response_data)


# --- 헬스 체크 엔드포인트 ---
@app.get("/health")
async def health_check():
    """서버 및 모델 로드 상태 확인"""
    return {
        "status": "healthy",
        "model_loaded": yolo is not None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- 루트 엔드포인트 ---
@app.get("/")
async def root():
    """API 기본 정보 반환"""
    return {
        "message": "백령도 해안 경계 AI 시스템 API",
        "docs_url": "/docs",
        "health_check": "/health"
    }

# 로컬에서 직접 실행 시 (예: python main_api_with_logging.py)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)