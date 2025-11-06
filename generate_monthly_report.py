import pandas as pd
import json
from collections import Counter
from datetime import datetime

LOG_FILE = 'detections.log'

def load_detection_logs(log_file: str) -> pd.DataFrame:
    """detections.log 파일을 읽어 Pandas DataFrame으로 변환"""
    logs = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    print(f"경고: 잘못된 형식의 로그 라인 발견, 건너뜁니다: {line}")
        
        if not logs:
            return pd.DataFrame() # 빈 DataFrame 반환
            
        df = pd.DataFrame(logs)
        # timestamp 열을 datetime 객체로 변환
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    except FileNotFoundError:
        print(f"오류: 로그 파일을 찾을 수 없습니다. ({log_file})")
        return pd.DataFrame()
    except Exception as e:
        print(f"오류: 로그 파일 로딩 중 문제 발생: {e}")
        return pd.DataFrame()

def generate_report(df: pd.DataFrame, year: int, month: int):
    """지정된 연도와 월의 통계 리포트를 생성"""
    
    # 해당 월의 데이터만 필터링
    df_month = df[
        (df['timestamp'].dt.year == year) & (df['timestamp'].dt.month == month)
    ].copy()

    if df_month.empty:
        print(f"=== {year}년 {month}월 탐지 리포트 ===")
        print("데이터가 없습니다.")
        print("="*40)
        return

    print(f"=== {year}년 {month}월 탐지 리포트 ===")
    print(f"총 탐지 건수 (주의 이상): {len(df_month)} 건")
    print("\n" + "-"*20)
    
    # 1. 레벨별 통계
    level_counts = df_month['level'].value_counts()
    print("[레벨별 통계]")
    for level, count in level_counts.items():
        print(f"- {level}: {count} 건")

    print("\n" + "-"*20)

    # 2. 탐지 객체별 통계 (복수 객체 처리)
    # detected_objects는 리스트 형태 (예: ["어선 → 중간 거리", "사람 → 매우 가까움"])
    # 리스트에서 객체 이름만 추출
    object_counter = Counter()
    
    def extract_objects(obj_list):
        if not isinstance(obj_list, list):
            return
        for item in obj_list:
            # "어선 → 중간 거리" 에서 "어선"만 추출
            obj_name = item.split('→')[0].strip()
            object_counter[obj_name] += 1

    df_month['detected_objects'].apply(extract_objects)

    print("[탐지된 주요 객체 통계]")
    if not object_counter:
        print("탐지된 객체 정보가 없습니다.")
    else:
        # 가장 많이 탐지된 순서대로 정렬
        for obj_name, count in object_counter.most_common():
            print(f"- {obj_name}: {count} 건")
    
    print("\n" + "="*40)


if __name__ == "__main__":
    # 로그 파일 로드
    df_logs = load_detection_logs(LOG_FILE)
    
    if not df_logs.empty:
        # 현재 날짜를 기준으로 '지난 달' 리포트를 생성
        today = datetime.today()
        first_day_of_month = today.replace(day=1)
        last_day_of_last_month = first_day_of_month - pd.Timedelta(days=1)
        
        report_year = last_day_of_last_month.year
        report_month = last_day_of_last_month.month
        
        print(f"'{LOG_FILE}' 파일을 기반으로 {report_year}년 {report_month}월 리포트를 생성합니다.\n")
        generate_report(df_logs, report_year, report_month)
        
        # (참고) 현재 월 리포트 생성
        # --- 수정된 부분: 아래 두 줄의 주석을 제거 ---
        print("\n참고: 현재 월 리포트")
        generate_report(df_logs, today.year, today.month)
        
    else:
        print(f"'{LOG_FILE}'에 분석할 데이터가 없습니다.")