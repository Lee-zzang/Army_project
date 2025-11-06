import os
import json
import shutil
import yaml
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ==================== 설정 ====================
class_map = {
    "어선": 0,
    "상선": 1,
    "군함": 2,
    "사람": 3,
    "유조류": 4
}

# ==================== JSON → YOLO 변환 ====================
def convert_single_file(args):
    """단일 JSON 파일을 YOLO 포맷으로 변환"""
    file, json_dir, img_dir, out_dir = args
    
    result = {"status": "success", "file": file, "lines": 0}
    
    try:
        json_path = os.path.join(json_dir, file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if not anns:
            result["status"] = "skipped"
            result["reason"] = "no_annotations"
            return result

        yolo_lines = []
        for ann in anns:
            # 클래스 ID 검증
            try:
                cls_id = int(ann["class"]) - 1  # 1~5 → 0~4
            except (KeyError, ValueError):
                result["status"] = "error"
                result["reason"] = f"invalid_class: {ann.get('class', 'N/A')}"
                return result
            
            if not 0 <= cls_id <= 4:
                result["status"] = "error"
                result["reason"] = f"class_out_of_range: {cls_id}"
                return result

            # 이미지 크기 가져오기
            img_path = os.path.join(img_dir, ann["filename"])
            if not os.path.exists(img_path):
                result["status"] = "skipped"
                result["reason"] = f"image_not_found: {ann['filename']}"
                return result

            try:
                with Image.open(img_path) as im:
                    img_w, img_h = im.size
            except Exception as e:
                result["status"] = "error"
                result["reason"] = f"image_read_error: {e}"
                return result

            # 바운딩 박스 좌표 변환
            x, y, w, h = ann["bbox"]
            
            # YOLO 포맷: 중심점 기준 정규화
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h
            
            # 좌표 범위 검증 (0~1)
            if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= w_norm <= 1 and 0 <= h_norm <= 1):
                result["status"] = "error"
                result["reason"] = f"invalid_bbox: ({x_center:.2f}, {y_center:.2f}, {w_norm:.2f}, {h_norm:.2f})"
                return result

            yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # YOLO 라벨 파일 저장
        if yolo_lines:
            txt_name = file.replace(".json", ".txt")
            txt_path = os.path.join(out_dir, txt_name)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines))
            
            result["lines"] = len(yolo_lines)
        else:
            result["status"] = "skipped"
            result["reason"] = "no_valid_boxes"
    
    except Exception as e:
        result["status"] = "error"
        result["reason"] = str(e)
    
    return result


def convert_json_to_yolo(json_dir, img_dir, out_dir, workers=4, verbose=True):
    """JSON 라벨 → YOLO txt 변환 (병렬처리 + 통계)"""
    os.makedirs(out_dir, exist_ok=True)
    
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]
    
    if not json_files:
        print(f"❌ {json_dir}에 JSON 파일이 없습니다.")
        return {"converted": 0, "skipped": 0, "errors": 0}
    
    print(f"\n🔄 [1단계] JSON → YOLO 변환 시작: {len(json_files)}개 파일")
    
    stats = {
        "converted": 0,
        "skipped": 0,
        "errors": 0,
        "total_boxes": 0,
        "error_details": []
    }
    
    args_list = [(f, json_dir, img_dir, out_dir) for f in json_files]
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_single_file, args): args[0] for args in args_list}
        
        with tqdm(total=len(json_files), desc="   변환 중", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                
                if result["status"] == "success":
                    stats["converted"] += 1
                    stats["total_boxes"] += result["lines"]
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                    if verbose:
                        stats["error_details"].append(f"⚠️ {result['file']}: {result['reason']}")
                else:
                    stats["errors"] += 1
                    stats["error_details"].append(f"❌ {result['file']}: {result['reason']}")
                
                pbar.update(1)
    
    print(f"   ✅ 성공: {stats['converted']}개 ({stats['total_boxes']}개 박스)")
    print(f"   ⚠️ 스킵: {stats['skipped']}개")
    print(f"   ❌ 오류: {stats['errors']}개")
    
    return stats


# ==================== 데이터셋 필터링 ====================
def filter_dataset(img_dir, lbl_dir, out_img, out_lbl, split_name=""):
    """이미지-라벨 매칭 검증 후 필터링"""
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    
    copied = 0
    skipped = 0
    
    for lbl in tqdm(label_files, desc=f"   {split_name} 필터링", unit="file"):
        base = os.path.splitext(lbl)[0]
        
        # 다양한 이미지 확장자 지원
        img_file = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            candidate = os.path.join(img_dir, base + ext)
            if os.path.exists(candidate):
                img_file = candidate
                break
        
        lbl_file = os.path.join(lbl_dir, lbl)
        
        if img_file and os.path.exists(lbl_file):
            # 라벨 파일이 비어있는지 확인
            if os.path.getsize(lbl_file) > 0:
                shutil.copy(img_file, out_img)
                shutil.copy(lbl_file, out_lbl)
                copied += 1
            else:
                skipped += 1
        else:
            skipped += 1
    
    return copied, skipped


# ==================== YAML 생성 ====================
def create_data_yaml(base_dir, output_path, train_path, val_path, test_path=None):
    """YOLO 학습용 data.yaml 생성"""
    data_yaml = {
        "path": base_dir.replace("\\", "/"),
        "train": train_path.replace("\\", "/"),
        "val": val_path.replace("\\", "/"),
        "nc": 5,
        "names": ["어선", "상선", "군함", "사람", "유조류"]
    }
    
    if test_path:
        data_yaml["test"] = test_path.replace("\\", "/")
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, allow_unicode=True, default_flow_style=False)
    
    print(f"   ✅ data.yaml 생성: {output_path}")


# ==================== 메인 파이프라인 ====================
def preprocess_army_dataset(base_dir, workers=8, skip_conversion=False):
    """
    전체 데이터 전처리 파이프라인
    
    Args:
        base_dir: 데이터셋 루트 디렉토리
        workers: 병렬 처리 워커 수
        skip_conversion: JSON 변환 건너뛰기 (이미 변환된 경우)
    """
    print("╔═══════════════════════════════════════╗")
    print("║   국방 AI 데이터 전처리 시스템      ║")
    print("╚═══════════════════════════════════════╝")
    
    # 경로 설정
    base_path = Path(base_dir)
    
    # 원본 경로
    train_json_dir = base_path / "Train" / "json"
    train_img_dir = base_path / "Train" / "Origin"
    train_lbl_dir = base_path / "Train" / "labels"
    
    val_json_dir = base_path / "Val" / "json"
    val_img_dir = base_path / "Val" / "Origin"
    val_lbl_dir = base_path / "Val" / "labels"
    
    # 필터링된 데이터 경로
    filtered_base = base_path / "Filtered"
    train_out_img = filtered_base / "Train" / "images"
    train_out_lbl = filtered_base / "Train" / "labels"
    val_out_img = filtered_base / "Val" / "images"
    val_out_lbl = filtered_base / "Val" / "labels"
    
    # ========== 1단계: JSON → YOLO 변환 ==========
    if not skip_conversion:
        # Train 변환
        if train_json_dir.exists():
            train_stats = convert_json_to_yolo(
                str(train_json_dir),
                str(train_img_dir),
                str(train_lbl_dir),
                workers=workers
            )
        
        # Val 변환
        if val_json_dir.exists():
            val_stats = convert_json_to_yolo(
                str(val_json_dir),
                str(val_img_dir),
                str(val_lbl_dir),
                workers=workers
            )
    else:
        print("\n⏭️  [1단계] JSON 변환 건너뛰기 (skip_conversion=True)")
    
    # ========== 2단계: 데이터셋 필터링 ==========
    print("\n🔍 [2단계] 데이터셋 필터링 시작")
    
    train_copied, train_skipped = filter_dataset(
        str(train_img_dir),
        str(train_lbl_dir),
        str(train_out_img),
        str(train_out_lbl),
        split_name="Train"
    )
    
    val_copied, val_skipped = filter_dataset(
        str(val_img_dir),
        str(val_lbl_dir),
        str(val_out_img),
        str(val_out_lbl),
        split_name="Val"
    )
    
    print(f"   ✅ Train: {train_copied}개 복사, {train_skipped}개 스킵")
    print(f"   ✅ Val: {val_copied}개 복사, {val_skipped}개 스킵")
    
    # ========== 3단계: data.yaml 생성 ==========
    print("\n📝 [3단계] YOLO 학습 설정 파일 생성")
    
    yaml_path = base_path / "data_filtered.yaml"
    create_data_yaml(
        base_dir=str(filtered_base),
        output_path=str(yaml_path),
        train_path="Train/images",
        val_path="Val/images"
    )
    
    # ========== 최종 리포트 ==========
    print("\n" + "="*50)
    print("✅ 데이터 전처리 완료!")
    print("="*50)
    print(f"📊 최종 데이터셋:")
    print(f"   - Train: {train_copied}개")
    print(f"   - Val: {val_copied}개")
    print(f"   - 총합: {train_copied + val_copied}개")
    print(f"\n📂 출력 디렉토리: {filtered_base}")
    print(f"📄 학습 설정: {yaml_path}")
    print("\n🚀 다음 단계: YOLO 모델 학습")
    print(f"   python train.py --data {yaml_path} --epochs 100")


# ==================== 실행 예시 ====================
if __name__ == "__main__":
    # 데이터셋 경로 설정
    BASE_DIR = r"C:/Army_project/data"
    
    # 전체 파이프라인 실행
    preprocess_army_dataset(
        base_dir=BASE_DIR,
        workers=8,              # CPU 코어 수에 맞게 조정
        skip_conversion=False   # JSON 변환 건너뛰기 (False = 변환 수행)
    )