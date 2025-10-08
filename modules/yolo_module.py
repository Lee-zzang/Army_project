from ultralytics import YOLO
import cv2

# 학습된 YOLO 모델 로드
model = YOLO("runs/army_project_clean_yolo11s/weights/best.pt")

def detect_objects(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {img_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img, conf=0.25)

    detected_objects = []
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

    return results, detected_objects
