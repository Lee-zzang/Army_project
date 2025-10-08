import os
import shutil
import yaml

# 원본 데이터 경로
base_dir = r"C:/Army_project/data"
train_img_dir = os.path.join(base_dir, "Train/Origin")
train_lbl_dir = os.path.join(base_dir, "Train/labels")
val_img_dir = os.path.join(base_dir, "Val/Origin")
val_lbl_dir = os.path.join(base_dir, "Val/labels")

# 새로운 필터링된 데이터셋 경로
out_base = os.path.join(base_dir, "Filtered")
train_out_img = os.path.join(out_base, "Train/Origin")
train_out_lbl = os.path.join(out_base, "Train/labels")
val_out_img = os.path.join(out_base, "Val/Origin")
val_out_lbl = os.path.join(out_base, "Val/labels")

# 폴더 생성
for d in [train_out_img, train_out_lbl, val_out_img, val_out_lbl]:
    os.makedirs(d, exist_ok=True)

def filter_dataset(img_dir, lbl_dir, out_img, out_lbl):
    copied = 0
    for lbl in os.listdir(lbl_dir):
        if lbl.endswith(".txt"):
            base = os.path.splitext(lbl)[0]
            img_file = os.path.join(img_dir, base + ".jpg")
            lbl_file = os.path.join(lbl_dir, lbl)
            if os.path.exists(img_file):
                shutil.copy(img_file, out_img)
                shutil.copy(lbl_file, out_lbl)
                copied += 1
    return copied

# Train/Val 필터링
train_copied = filter_dataset(train_img_dir, train_lbl_dir, train_out_img, train_out_lbl)
val_copied = filter_dataset(val_img_dir, val_lbl_dir, val_out_img, val_out_lbl)

print(f"✅ Train 데이터 복사 완료: {train_copied}개")
print(f"✅ Val 데이터 복사 완료: {val_copied}개")

# 새로운 data.yaml 생성
data_yaml = {
    "train": os.path.join(out_base, "Train/Origin").replace("\\", "/"),
    "val": os.path.join(out_base, "Val/Origin").replace("\\", "/"),
    "nc": 5,
    "names": ["어선", "상선", "군함", "사람", "유조류"]
}

out_yaml_path = os.path.join(base_dir, "data_filtered.yaml")
with open(out_yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(data_yaml, f, allow_unicode=True)

print(f"✅ 새로운 data.yaml 생성 완료: {out_yaml_path}")
