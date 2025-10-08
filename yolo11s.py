from ultralytics import YOLO

# 마지막 체크포인트 불러오기
model = YOLO("runs/army_project_clean_yolo11s/weights/last.pt")

# 이어서 학습
results = model.train(
    data=r"C:\Army_project\data_filtered.yaml",
    epochs=3,             # 이어서 더 학습할 epoch 수
    imgsz=640,
    batch=16,
    resume=True           # ✅ 이어서 학습 옵션 추가
)

print("✅ 훈련 완료! 결과 저장 위치:", results.save_dir)