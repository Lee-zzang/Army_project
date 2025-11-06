import streamlit as st
import os
import glob
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# -----------------------------------------------------------------
# 1. 함수 정의를 파일 상단 (사용되는 곳보다 위)으로 이동
# -----------------------------------------------------------------

def process_image(image_data: (str | bytes), api_url: str, result_column, filename: str = None):
    """이미지 처리 및 결과 표시 (파일 경로 또는 바이트 데이터 처리)"""
    
    files = None
    if isinstance(image_data, str):
        try:
            f = open(image_data, "rb")
            files = {"file": (os.path.basename(image_data), f, "image/jpeg")}
        except Exception as e:
            st.error(f"이미지 파일을 열 수 없습니다: {e}")
            return
    elif isinstance(image_data, bytes):
        if filename is None: filename = "uploaded_image.jpg"
        files = {"file": (filename, image_data, "image/jpeg")}
    
    if not files:
        st.error("처리할 이미지가 없습니다.")
        return

    with st.spinner("YOLO 탐지 및 경고(TTS) 생성 중..."):
        try:
            response = requests.post(
                f"{api_url}/detect",
                files=files,
                timeout=30 # TTS 생성 시간을 고려해 timeout 넉넉하게
            )
            
            if response.status_code == 200:
                result = response.json()
                display_results(result, result_column)
            else:
                st.error(f"API 오류: {response.status_code} - {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error(f"FastAPI 서버에 연결할 수 없습니다: {api_url}")
            st.info("서버가 실행 중인지 확인하세요.")
        except Exception as e:
            st.error(f"오류 발생: {str(e)}")
        finally:
            if isinstance(image_data, str) and 'f' in locals():
                f.close()


def display_results(result: dict, result_column):
    """탐지 결과 표시 (동적 TTS 재생 기능으로 수정)"""
    
    with result_column:
        st.markdown("#### 탐지 결과")
        # 'result.get'으로 'detections' 키를 안전하게 가져옵니다.
        detections = result.get("detections", [])
        
        if detections:
            # 3. 'deteCTIONS' 오타 수정 -> 'detections' (소문자)
            st.markdown(f"**탐지된 객체: {len(detections)}개**")
            
            for i, det in enumerate(detections, 1):
                with st.expander(f"{i}. {det['class_name']} ({det['confidence']})", expanded=True):
                    cols = st.columns([1, 1])
                    with cols[0]: st.metric("신뢰도", f"{det['confidence']}")
                    with cols[1]: st.metric("거리", det['distance_status'])
        else:
            st.info("탐지된 객체가 없습니다.")
    
    st.markdown("---")
    st.subheader("전술 경고 메시지")
    
    warning = result.get("warning", {})
    level = warning.get("level", "안전")
    
    # --- 1. 텍스트 경고 표시 (이모지 제거) ---
    if level == "경보":
        st.error(f"**[{level}]** {warning.get('summary', '')}")
    elif level == "주의":
        st.warning(f"**[{level}]** {warning.get('summary', '')}")
    else:
        st.success(f"**[{level}]** {warning.get('summary', '')}")
    
    st.markdown(f"**권장 조치:** {warning.get('action', '')}")
    
    # --- 2. 동적 TTS 음성 재생 (신규) ---
    audio_base64 = warning.get("audio_base64")
    if audio_base64:
        audio_html = f"""
        <audio autoplay="true">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.components.v1.html(audio_html, height=0)
    
    # --- 3. 상세 정보 표시 (이모지 제거) ---
    with st.expander("상세 정보"):
        st.json(result)

# -----------------------------------------------------------------
# 2. Streamlit UI 코드 시작 (함수 정의 이후)
# -----------------------------------------------------------------

st.set_page_config(
    page_title="백령도 해안 경계 AI 시스템 (TTS)",
    page_icon=None,
    layout="wide"
)

# [핵심] st.sidebar, st.title 등 UI 요소 정의
st.sidebar.title("메뉴") 
st.title("백령도 해안 경계 AI 시스템 (TTS)")

# [핵심] 2. 'tab1' 오류 해결: 주석(#)을 제거하여 tab1, tab2 변수를 정의합니다.
tab1, tab2 = st.tabs(["로컬 이미지 선택", "이미지 업로드"])

# (탭 1: 로컬 이미지 선택)
# 이제 tab1이 정의되었으므로 이 코드는 정상입니다.
with tab1:
    st.subheader("드론 촬영 이미지 선택")
    
    test_dir = st.text_input(
        "이미지 폴더 경로",
        value=r"C:\Army_project\data\Filtered\Train\images", # 예시 경로
        help="드론 촬영 이미지가 저장된 폴더"
    )
    
    if os.path.exists(test_dir):
        image_files = glob.glob(os.path.join(test_dir, "*.jpg")) + \
                      glob.glob(os.path.join(test_dir, "*.png"))
        
        if not image_files:
            st.warning("해당 폴더에 이미지가 없습니다.")
        else:
            st.success(f"{len(image_files)}개의 이미지를 찾았습니다.")
            
            selected_img = st.selectbox(
                "이미지 선택",
                image_files,
                format_func=lambda x: os.path.basename(x)
            )
            
            if selected_img:
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("#### 원본 이미지")
                    try:
                        img = Image.open(selected_img)
                        st.image(img, use_column_width=True)
                    except Exception as e:
                        st.error(f"이미지 로드 실패: {e}")
                
                if st.button("객체 탐지 실행", type="primary", use_container_width=True):
                    api_url = "http://localhost:8000" # 임시 하드코딩
                    # 이제 process_image 함수가 위쪽에 정의되어 있으므로 정상입니다.
                    process_image(selected_img, api_url, col2)
    else:
        st.error(f"폴더를 찾을 수 없습니다: {test_dir}")

# (탭 2: 이미지 업로드)
with tab2:
    st.subheader("이미지 업로드")
    
    uploaded_file = st.file_uploader(
        "드론 촬영 이미지를 업로드하세요",
        type=["jpg", "jpeg", "png"],
        help="JPG, PNG 형식 지원"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### 업로드된 이미지")
            img = Image.open(uploaded_file)
            st.image(img, use_column_width=True)
        
        if st.button("객체 탐지 실행", type="primary", use_container_width=True, key="upload_detect"):
            img_bytes = uploaded_file.getvalue()
            api_url = "http://localhost:8000" # 임시 하드코딩
            # 이제 process_image 함수가 위쪽에 정의되어 있으므로 정상입니다.
            process_image(img_bytes, api_url, col2, filename=uploaded_file.name)