import streamlit as st
import os, glob
from modules.yolo_module import detect_objects
from modules.llm_module import generate_warning

st.set_page_config(page_title="드론 AI Agent", layout="wide")
st.title("🚨 드론 객체 탐지 & 경고 메시지 시스템")

# 드론 데이터 (Test 폴더 자동 불러오기)
test_dir = r"C:\Army_project\data\Filtered\Train\images"
image_files = glob.glob(os.path.join(test_dir, "*.jpg"))

if not image_files:
    st.warning("⚠️ Test 폴더에 이미지가 없습니다.")
else:
    selected_img = st.selectbox("드론 촬영 이미지 선택", image_files)

    with st.spinner("YOLO 탐지 중..."):
        results, detected_objects = detect_objects(selected_img)

    with st.spinner("LLM 메시지 생성 중..."):
        warning_msg = generate_warning(detected_objects)

    # 출력
    st.subheader("📷 탐지 결과")
    st.image(results[0].plot(), caption=os.path.basename(selected_img), use_column_width=True)

    st.subheader("🚨 경고 메시지")
    st.info(warning_msg)
