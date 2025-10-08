import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_warning(detected_objects):
    """
    detected_objects: ["person → ⚠️ 매우 가까움", "car → ✔️ 멀리 있음", ...]
    return: 최종 경고 메시지 (문장)
    """
    if not detected_objects:
        return "탐지된 객체 없음"

    # 입력을 보기 좋게 정리
    items = "\n".join(f"- {x}" for x in detected_objects)

    # 🔹 Few-shot + 단계별 역할 지시 (출력은 '최종 메시지'만)
    messages = [
        {
            "role": "system",
            "content": (
                "너는 '전술 경고 시스템'이다. 출력은 항상 한국어로 하며, "
                "내부적으로는 다음 3단계를 순서대로 수행하되 최종 결과만 출력한다. "
                "1) 상황요약관: 탐지 결과를 분류·중복 제거하고 가장 위험한 항목을 식별한다. "
                "2) 위험도평가관: 가까움/중간/멀리 있음 규칙을 바탕으로 전체 심각도를 산출한다. "
                "3) 통신장교: 짧고 명확한 경고 메시지를 작성한다. "
                "규칙: 과장하거나 존재하지 않는 객체를 추가하지 말 것. 민간/우군 가능성은 모호하면 단정하지 말 것."
            ),
        },
        {
            "role": "system",
            "content": (
                "출력 형식은 다음 가이드를 따른다.\n"
                "[경보|주의|안전] 한 줄 요약\n"
                "조치: 즉시 취해야 할 1~2가지 행동\n"
                "예) [경보] 인원이 매우 근접해 접근 중.\n"
                "조치: 즉시 후퇴 및 시야 확보."
            ),
        },

        # 🔹 Few-shot 예시 1
        {"role": "user", "content": "탐지 결과:\n- person → ⚠️ 매우 가까움"},
        {
            "role": "assistant",
            "content": (
                "[경보] 인원이 바로 앞에서 관측됩니다.\n"
                "조치: 즉시 엄폐 확보, 후퇴하면서 추가 관측."
            ),
        },

        # 🔹 Few-shot 예시 2
        {"role": "user", "content": "탐지 결과:\n- car → ✔️ 멀리 있음\n- person → ⚠️ 중간 거리"},
        {
            "role": "assistant",
            "content": (
                "[주의] 인원이 중간 거리에서 관측됩니다. 차량은 멀리 있습니다.\n"
                "조치: 관측 유지, 접근 시 경고 방송 준비."
            ),
        },

        # 🔹 실제 입력
        {"role": "user", "content": f"탐지 결과:\n{items}"},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,  # 경고문은 보수적으로
    )
    return resp.choices[0].message.content
