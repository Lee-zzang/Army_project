import os
import json
from PIL import Image

class_map = {
    "어선": 0,
    "상선": 1,
    "군함": 2,
    "사람": 3,
    "유조류": 4
}

def convert_json_to_yolo(json_dir, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for file in os.listdir(json_dir):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(json_dir, file), "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if not anns:
            continue

        yolo_lines = []
        for ann in anns:
            # 이미지 크기 가져오기 (JSON이 0이면 이미지 직접 읽기)
            img_path = os.path.join(img_dir, ann["filename"])
            if os.path.exists(img_path):
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            else:
                print(f"⚠️ 이미지 없음: {ann['filename']}, 스킵")
                continue

            cls_id = int(ann["class"]) - 1  # 1~5 → 0~4
            x, y, w, h = ann["bbox"]

            # YOLO 포맷
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w /= img_w
            h /= img_h

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            txt_name = file.replace(".json", ".txt")
            with open(os.path.join(out_dir, txt_name), "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))

    print(f"✅ {json_dir} → {out_dir} 변환 완료")
