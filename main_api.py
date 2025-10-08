from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from modules.llm_module import generate_warning
import cv2
import numpy as np

app = FastAPI(title="Drone AI Agent")

# ✅ YOLO 모델 로드 (서버 시작 시 1번만)
yolo = None

@app.on_event("startup")
def load_model():
    global yolo
    try:
        yolo = YOLO("runs/army_project_clean_yolo11s/weights/best.pt")
    except Exception:
        yolo = YOLO("yolov8n.pt")  # 백업용

# ✅ /detect 엔드포인트
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_img = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("이미지 디코딩 실패")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 열 수 없습니다.")

    # YOLO 탐지 실행
    results = yolo.predict(img, verbose=False)
    detected_objects = []
    if results and results[0].boxes is not None:
        img_h, img_w = results[0].orig_shape
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = results[0].names[cls_id]
            xyxy = box.xyxy[0].tolist()
            h = xyxy[3] - xyxy[1]
            ratio = h / img_h

            if ratio > 0.5:
                status = "⚠️ 매우 가까움"
            elif ratio > 0.2:
                status = "⚠️ 중간 거리"
            else:
                status = "✔️ 멀리 있음"

            detected_objects.append(f"{label} → {status}")

    # LLM 기반 경고 메시지
    warning_msg = generate_warning(detected_objects)

    return JSONResponse(content={
        "detected_objects": detected_objects,
        "warning_message": warning_msg
    })
