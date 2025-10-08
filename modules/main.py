from modules.yolo_module import detect_objects
from modules.llm_module import generate_warning

if __name__ == "__main__":
    # 테스트 이미지 경로
    test_img = r"data/Filtered/Train/images/I2_S0_C5_0008068.jpg"

    # 1) YOLO 탐지
    results, objects = detect_objects(test_img)
    print("=== 탐지 결과 ===")
    for obj in objects:
        print(obj)

    # 2) LLM 기반 경고 메시지
    warning_msg = generate_warning(objects)
    print("\n=== 경고 메시지 ===")
    print(warning_msg)
