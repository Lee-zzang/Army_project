# 이미지와 라벨이 있는지 확인하는 코드

import os

val_img_dir = r"C:\Army_project\data\Filtered\Val\images"
val_label_dir = r"C:\Army_project\data\Filtered\Val\labels"

images = [f for f in os.listdir(val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
labels = [f for f in os.listdir(val_label_dir) if f.endswith('.txt')]

image_names = set(os.path.splitext(f)[0] for f in images)
label_names = set(os.path.splitext(f)[0] for f in labels)

print("✅ 총 이미지 개수:", len(images))
print("✅ 총 라벨 개수:", len(labels))

missing_labels = image_names - label_names
missing_images = label_names - image_names

if missing_labels:
    print("❌ 라벨 없는 이미지:", list(missing_labels)[:10])  # 일부만 출력
if missing_images:
    print("❌ 이미지 없는 라벨:", list(missing_images)[:10])

if not missing_labels and not missing_images:
    print("🎉 이미지와 라벨이 완벽히 매칭됩니다.")
