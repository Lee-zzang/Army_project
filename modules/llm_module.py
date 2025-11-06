import os
import json
import logging
from datetime import datetime
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import base64 # 음성 데이터 처리를 위해 base64 추가

# --- 환경 설정 ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def _generate_tts_audio(text_to_speak: str) -> (str or None):
    """
    주어진 텍스트를 OpenAI TTS-1을 사용해 MP3 음성으로 변환하고
    Base64 문자열로 반환합니다.
    """
    try:
        response = client.audio.speech.create(
            model="tts-1",      # 빠르고 품질 좋은 모델
            voice="nova",       # 선명한 여성 목소리 (한국어 지원)
            input=text_to_speak
        )
        # 응답 받은 오디오 바이트를 Base64로 인코딩
        audio_bytes = response.content
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        logging.info("TTS 음성 생성 성공")
        return audio_base64
    except Exception as e:
        logging.warning(f"TTS 음성 생성 실패: {e}")
        return None

def generate_fallback_warning(detected_objects: List[str]) -> Dict[str, str]:
    """해안 경계용 규칙 기반 경고 생성 (TTS 기능 추가)"""
    # ... (기존 fallback 로직과 동일) ...
    critical = [obj for obj in detected_objects if "매우 가까움" in obj]
    warning = [obj for obj in detected_objects if "중간 거리" in obj]
    
    high_risk_objects = ["사람", "어선", "군함"]
    has_high_risk = any(
        any(risk in obj for risk in high_risk_objects) 
        for obj in critical
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. 반환할 딕셔너리 생성
    if critical and has_high_risk:
        result_dict = {
            "level": "경보",
            "summary": f"해안가 근접 객체 탐지 ({len(critical)}개)",
            "action": "즉시 육안 확인 및 상급부대 보고",
            "source": "fallback"
        }
    elif warning:
        result_dict = {
            "level": "주의",
            "summary": f"선박 또는 인원 관측 ({len(warning)}개)",
            "action": "이동 경로 지속 관측",
            "source": "fallback"
        }
    else:
        result_dict = {
            "level": "안전",
            "summary": "원거리 객체만 관측됨",
            "action": "정상 경계 유지",
            "source": "fallback"
        }

    result_dict["raw_detections"] = detected_objects
    result_dict["timestamp"] = timestamp
    
    # 2. TTS 생성
    text_to_speak = f"[{result_dict['level']}] {result_dict['summary']}"
    result_dict["audio_base64"] = _generate_tts_audio(text_to_speak)
    
    return result_dict


def generate_warning(detected_objects: List[str], max_retries: int = 3) -> Dict[str, str]:
    """
    YOLO 탐지 결과를 자연어 경고 메시지 및 TTS 음성으로 변환 (수정)
    """
    start_time = datetime.now()
    
    # ... (빈 입력 처리 로직은 동일) ...
    if not detected_objects:
        empty_result = {
            "level": "안전",
            "summary": "탐지된 객체 없음",
            "action": "정상 경계 유지",
            "raw_detections": [],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "empty"
        }
        # TTS 생성
        text_to_speak = "[안전] 탐지된 객체 없음"
        empty_result["audio_base64"] = _generate_tts_audio(text_to_speak)
        return empty_result

    items = "\n".join(f"- {x}" for x in detected_objects)

    # ... (프롬프트 구성은 동일) ...
    messages = [
        {
            "role": "system",
            "content": (
                "너는 백령도 해안 경계 전술 경고 시스템이다. 출력은 반드시 다음 JSON 형식을 따른다:\n"
                "{\n"
                '  "level": "경보|주의|안전",\n'
                '  "summary": "한 줄 상황 요약",\n'
                '  "action": "즉시 취할 행동"\n'
                "}\n\n"
                "탐지 가능 객체: 어선, 상선, 군함, 사람, 유조류\n\n"
                "내부 처리 단계:\n"
                "1) 상황요약관: 탐지 결과 분류, 중복 제거, 위험 항목 식별\n"
                "2) 위험도평가관: 거리 기준으로 심각도 산출\n"
                "3) 통신장교: 해안 경계 상황에 맞는 짧고 명확한 경고 작성\n\n"
                "규칙: 과장 금지, 존재하지 않는 객체 추가 금지, 민간 선박/우군 판단은 신중히"
            ),
        },
        # ... (Few-shot 예시들은 동일) ...
        {
            "role": "user",
            "content": "탐지 결과:\n- 사람 → 매우 가까움"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "level": "경보",
                "summary": "해안가에 인원 근접 탐지",
                "action": "즉시 육안 확인, 상급부대 보고 준비"
            }, ensure_ascii=False)
        },
        {
            "role": "user",
            "content": "탐지 결과:\n- 어선 → 중간 거리\n- 상선 → 멀리 있음"
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "level": "주의",
                "summary": "어선이 중간 거리에서 관측됨. 상선은 멀리 있음",
                "action": "어선 이동 경로 지속 관측, 접근 시 식별 절차"
            }, ensure_ascii=False)
        },
        # ... (다른 예시들) ...
        {
            "role": "user",
            "content": f"탐지 결과:\n{items}"
        },
    ]

    # --- API 호출 (재시도 로직) ---
    for attempt in range(max_retries):
        try:
            # 1. LLM 텍스트 생성
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                timeout=10,
                response_format={"type": "json_object"}
            )
            
            # 2. 응답 파싱
            result = json.loads(resp.choices[0].message.content)
            result["raw_detections"] = detected_objects
            result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result["source"] = "llm"
            
            # --- 3. TTS 음성 생성 (신규 추가) ---
            # 관측병에게는 '요약'과 '조치'를 모두 들려주는 것이 좋습니다.
            text_to_speak = f"[{result['level']}] {result['summary']}. {result['action']}"
            result["audio_base64"] = _generate_tts_audio(text_to_speak)

            elapsed = (datetime.now() - start_time).total_seconds()
            logging.info(
                f"경고 생성 성공 (LLM+TTS) | 레벨: {result['level']} | "
                f"객체: {len(detected_objects)}개 | 소요시간: {elapsed:.2f}s"
            )
            
            return result
        
        except Exception as e:
            if attempt == max_retries - 1:
                logging.error(f"LLM 호출 최종 실패: {e} | 폴백 모드 전환")
                return generate_fallback_warning(detected_objects) # 폴백 함수도 TTS가 포함됨
            
            logging.warning(f"LLM 호출 실패 (시도 {attempt+1}/{max_retries}): {e}")
    
    return generate_fallback_warning(detected_objects) # 최종 폴백


def format_warning_text(warning: Dict[str, str]) -> str:
    """경고 메시지를 UI 표시용 텍스트로 변환 (동일)"""
    return f"[{warning['level']}] {warning['summary']}\n조치: {warning['action']}"

# ... (if __name__ == "__main__": 테스트 코드는 동일) ...