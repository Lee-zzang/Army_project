import os
import shutil
import random

def resample_train_to_balance(base_path, target_train=19520, target_val=4183, target_test=4183):
    """
    이미 Test 929개가 만들어진 상태에서, Train에서 일부를 떼어 Val/Test 보충
    """
    train_origin = os.path.join(base_path, "Train/Origin")
    train_label = os.path.join(base_path, "Train/Label")
    val_origin = os.path.join(base_path, "Val/Origin")
    val_label = os.path.join(base_path, "Val/Label")
    test_origin = os.path.join(base_path, "Test/Origin")
    test_label = os.path.join(base_path, "Test/Label")

    # 현재 개수
    cur_train = len(os.listdir(train_origin))
    cur_val = len(os.listdir(val_origin))
    cur_test = len(os.listdir(test_origin))
    total = cur_train + cur_val + cur_test

    print(f"현재: Train {cur_train}, Val {cur_val}, Test {cur_test}, 총 {total}")

    # 부족한 수 계산
    add_val = max(0, target_val - cur_val)
    add_test = max(0, target_test - cur_test)

    print(f"Train에서 Val로 {add_val}개, Test로 {add_test}개 이동 예정")

    # Train 이미지 무작위 선택
    train_images = [f for f in os.listdir(train_origin) if f.endswith((".jpg", ".png", ".jpeg"))]
    random.shuffle(train_images)

    # Val 보충
    for img in train_images[:add_val]:
        base = os.path.splitext(img)[0]
        lbl = base + ".json"
        shutil.move(os.path.join(train_origin, img), os.path.join(val_origin, img))
        shutil.move(os.path.join(train_label, lbl), os.path.join(val_label, lbl))

    # Test 보충
    for img in train_images[add_val:add_val+add_test]:
        base = os.path.splitext(img)[0]
        lbl = base + ".json"
        shutil.move(os.path.join(train_origin, img), os.path.join(test_origin, img))
        shutil.move(os.path.join(train_label, lbl), os.path.join(test_label, lbl))

    # 최종 결과 확인
    print(f"최종: Train {len(os.listdir(train_origin))}, "
          f"Val {len(os.listdir(val_origin))}, "
          f"Test {len(os.listdir(test_origin))}")

# 실행
base_path = r"C:\Army_project\data"
resample_train_to_balance(base_path)
