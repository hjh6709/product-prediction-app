import pandas as pd
from sqlalchemy import create_engine
import urllib
from tqdm import tqdm
import lightgbm as lgb
import numpy as np

# ✅ 1. DB 연결 설정 (이전과 동일)
server = 'localhost'
database = 'SPTEST1'
driver = 'ODBC Driver 17 for SQL Server'
params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# ✅ 2. 데이터 불러오기 (이전과 동일)
try:
    df = pd.read_sql("SELECT * FROM 출고테이블", engine)
    print("--- DB에서 데이터 로드 성공 ---")
except Exception as e:
    print(f"--- DB에서 데이터 로드 실패: {e} ---")
    print("DB 연결 문자열, 서버 상태, 데이터베이스/테이블 이름, ODBC 드라이버 설치를 확인하세요.")
    exit()

# --- DEBUG: 원본 df 정보 확인 (이전과 동일) ---
# ... (생략) ...
# -----------------------------------------------

# ✅ 3. 날짜 처리 및 월/연도/분기 피처 추출
temp_df = df.copy()
temp_df['출고일자_str'] = temp_df['출고일자'].astype(str)

try:
    temp_df['출고일자_parsed'] = pd.to_datetime(temp_df['출고년월_str'] + '01', format='%Y%m%d', errors='coerce')
    success_rate = temp_df['출고년월_parsed'].count() / len(temp_df)
    if success_rate > 0.5:
        df['출고년월'] = temp_df['출고년월_parsed']
        print("\n--- 날짜 변환 성공: 'YYYYMM' 형식으로 추정 ---")
    else:
        raise ValueError("YYYYMM 형식 변환 실패 또는 낮은 성공률")
except (ValueError, TypeError):
    print("\n--- 날짜 변환 실패: YYYYMM 형식 아님. 표준 날짜 형식 시도 ---")
    df['출고년월'] = pd.to_datetime(df['출고년월'], errors='coerce')

initial_rows = len(df)
df = df.dropna(subset=['출고년월'])
print(f"날짜 변환 실패로 제거된 행 수: {initial_rows - len(df)}")
print(f"결측치 제거 후 총 데이터 행 수: {len(df)}")

if len(df) == 0:
    print("\n❌ 날짜 변환 후 모든 데이터가 제거되었습니다. '출고년월' 컬럼의 실제 형식을 다시 확인해야 합니다. ❌")
    exit()

# --- ⭐ 새로운 피처: 시간 기반 피처 추가 (weekly_df 생성 전 df에 미리 추가) ⭐ ---
df['월'] = df['출고년월'].dt.month
df['년도'] = df['출고년월'].dt.year
df['분기'] = df['출고년월'].dt.quarter
df['월_시작_요일'] = df['출고년월'].dt.dayofweek
df['월_시작_연간주차'] = df['출고년월'].dt.isocalendar().week # 월의 시작일이 연간 몇번째 주차인지

# ✅ 4. 제품코드 + 월 단위 집계 (이전과 동일)
# 이제 df에 추가된 시간 피처들도 weekly_df에 자동으로 포함됩니다.
weekly_df = df.groupby(['제품코드', '출고년월', '년도', '월', '분기', '월_시작_요일', '월_시작_연간주차'])['확정수량'].sum().reset_index()
weekly_df = weekly_df.sort_values(by=['제품코드', '출고년월']).copy()
weekly_df['주차순서'] = weekly_df.groupby('제품코드').cumcount() # 이 주차순서는 여전히 월별 순서를 의미합니다.

# --- ⭐ 새로운 피처: 과거 값 (Lag Features) 및 이동 평균 추가 ⭐ ---
# '확정수량_직전월'은 예측 시점에는 알 수 없는 정보이므로, 예측에 사용하려면 주의가 필요합니다.
# 하지만 모델 학습 시에는 매우 유용합니다.
weekly_df['확정수량_직전월'] = weekly_df.groupby('제품코드')['확정수량'].shift(1)
weekly_df['확정수량_2개월전'] = weekly_df.groupby('제품코드')['확정수량'].shift(2)
# 1년 전 같은 월의 데이터는 계절성을 학습하는 데 매우 중요합니다.
weekly_df['확정수량_1년전'] = weekly_df.groupby('제품코드')['확정수량'].shift(12)
# 이동 평균
weekly_df['확정수량_3개월평균'] = weekly_df.groupby('제품코드')['확정수량'].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

# 새로운 피처들로 인해 NaN이 생길 수 있습니다. 모델 학습 전에 이들을 처리해야 합니다.
# 여기서는 가장 간단하게 0으로 채우지만, 데이터의 특성에 따라 평균값이나 다른 방식으로 채울 수 있습니다.
weekly_df.fillna(0, inplace=True) # NaN 값을 0으로 채우기 (간단한 처리)

print("\n--- weekly_df head (새로운 피처 포함) ---")
print(weekly_df.head())
print(f"월별 집계 후 총 데이터 행 수: {len(weekly_df)}")

if len(weekly_df) == 0:
    print("\n❌ 월별 집계 후 데이터가 없습니다. 제품코드 또는 확정수량에 문제가 있을 수 있습니다. ❌")
    exit()

print("--- 각 제품코드별 월별 데이터 개수 (상위 10개) ---")
product_counts = weekly_df['제품코드'].value_counts()
print(product_counts.head(10))


# ✅ 5. 예측 (LightGBM 사용)
results = []
codes = weekly_df['제품코드'].unique()
print(f"\n총 제품 수: {len(codes)}")

if len(codes) == 0:
    print("❌ 예측할 제품 코드가 없습니다.")
else:
    예측_시작_날짜 = pd.to_datetime('2025-08-01') # 모든 제품에 대해 2025년 8월 1일부터 예측 시작

    # LightGBM 모델 학습에 사용할 피처 리스트
    # '확정수량'과 '출고년월'은 타겟 및 시간 인덱스이므로 제외
    # '주차순서'는 그대로 사용, 새로 추가된 피처들 포함
    features = [
        '주차순서', '년도', '월', '분기', '월_시작_요일', '월_시작_연간주차',
        '확정수량_직전월', '확정수량_2개월전', '확정수량_1년전', '확정수량_3개월평균'
    ]

    for code in tqdm(codes, desc="월별 예측 중"):
        sub = weekly_df[weekly_df['제품코드'] == code].copy()

        # 학습을 위한 최소 데이터 (과거 값 피처로 인해 12개월 이상 필요할 수 있습니다)
        # Lag-12 피처를 사용하려면 최소 12개월 + 예측할 개월 수 이상의 데이터가 필요합니다.
        # 여기서는 최소 20개월 정도로 늘려보는 것을 권장합니다 (Lag-12 피처 사용 시).
        # 데이터가 충분치 않으면 Lag 피처를 뺀 후 진행하거나, min_periods를 낮춰야 합니다.
        if len(sub) < 20: # 최소 학습 개월 수 조정 (Lag-12를 사용한다면)
            # print(f"제품코드 {code}: 학습 데이터가 {len(sub)}개로 충분하지 않아 건너뜁니다.")
            continue

        # 학습 데이터에서 NaN이 없는 피처만 선택 (초반의 NaN은 fillna로 0 처리됨)
        X_train = sub[features]
        y_train = sub['확정수량']

        model = lgb.LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)

        # 예측할 미래 데이터프레임 생성
        future_data = []
        for i in range(4): # 다음 4개월 예측
            current_predicted_month_date = 예측_시작_날짜 + pd.DateOffset(months=i)
            
            # 예측 시점의 피처 값을 계산
            # '주차순서'는 학습된 마지막 순서 이후로 계속 증가
            # ⭐ 예측 시점의 과거 값 피처는 어떻게 채울 것인가? ⭐
            # 이 부분이 중요합니다. 예측 시점에는 실제 '직전월', '2개월전' 등의 값을 모릅니다.
            # 가장 간단한 방법: 마지막으로 관측된 값을 반복하거나 (forward fill), 0으로 채우거나,
            # 아니면 예측된 값으로 채워나가는 재귀적 예측 (recursive prediction)을 사용합니다.
            # 여기서는 마지막 관측된 값을 사용하는 간단한 방법을 채택합니다.
            
            # 미래 예측을 위한 주차순서
            future_order_for_model = sub['주차순서'].max() + i + 1

            # 미래 예측을 위한 피처 값
            # '확정수량_직전월' 등은 이전 단계에서 예측된 확정수량 값을 사용하거나, 마지막 학습값을 반복 사용
            # 여기서는 마지막 학습 데이터의 피처 값들을 기준으로 예측
            # 실제 예측에서는 좀 더 복잡한 로직이 필요할 수 있습니다. (예: 직전 예측값 사용 등)
            
            # 일단은 마지막 학습 시점의 피처 값들을 가져와서 새로운 '년도', '월' 등을 업데이트
            last_features_row = sub.iloc[-1]
            future_features = {
                '주차순서': future_order_for_model,
                '년도': current_predicted_month_date.year,
                '월': current_predicted_month_date.month,
                '분기': current_predicted_month_date.quarter,
                '월_시작_요일': current_predicted_month_date.dayofweek,
                '월_시작_연간주차': current_predicted_month_date.isocalendar().week,
                # ⭐ 과거 값 피처 처리: 가장 어려운 부분 ⭐
                # 여기서는 가장 최근의 관측된 값 (last_features_row)을 사용하거나,
                # 더 정교하게는, 이전 예측 단계에서 얻은 예측값을 활용하여 순차적으로 채워나갑니다.
                # 현재는 간단하게 마지막 학습 데이터의 값을 가져와서 사용
                '확정수량_직전월': last_features_row['확정수량'], # 직전 예측값으로 대체해야 정확
                '확정수량_2개월전': last_features_row['확정수량_직전월'], # 역시 직전 예측값의 과거
                '확정수량_1년전': last_features_row['확정수량_1년전'], # 1년 전 데이터는 예측 시점에도 동일한 달의 1년 전 값 사용
                '확정수량_3개월평균': last_features_row['확정수량_3개월평균'] # 역시 직전 예측값의 평균
            }
            future_data.append(future_features)
        
        future_df_for_prediction = pd.DataFrame(future_data, columns=features)
        
        # 모델 예측
        predicted_quantities = model.predict(future_df_for_prediction)
        
        # 결과를 저장할 데이터프레임 생성
        current_product_forecast = pd.DataFrame({
            '제품코드': code,
            '예측수량': predicted_quantities.round().astype(int)
        })
        current_product_forecast['예측수량'] = current_product_forecast['예측수량'].apply(lambda x: max(0, x))

        # ✅ 예측된 월 순서를 고정된 예측 날짜와 '연-월-주차'로 변환
        future_dates_info = []
        for i in range(4):
            current_predicted_month_date = 예측_시작_날짜 + pd.DateOffset(months=i)
            
            year_str = current_predicted_month_date.strftime('%Y년')
            month_str = current_predicted_month_date.strftime('%m월')
            
            days_in_month = pd.date_range(start=current_predicted_month_date, end=current_predicted_month_date + pd.offsets.MonthEnd(0))
            first_iso_week_of_month = days_in_month.isocalendar().week.min()
            
            current_iso_week = current_predicted_month_date.isocalendar().week
            week_of_month = current_iso_week - first_iso_week_of_month + 1
            if week_of_month <= 0:
                week_of_month = 1

            week_str = f'{int(week_of_month)}주차'

            future_dates_info.append({
                '예측_기간': f'{year_str} {month_str} {week_str}'
            })
        
        future_dates_df = pd.DataFrame(future_dates_info)
        current_product_forecast['예측_기간'] = future_dates_df['예측_기간']
        
        results.append(current_product_forecast)

# ✅ 6. 결과 출력
if results:
    forecast_df = pd.concat(results, ignore_index=True)
    print("\n--- 최종 예측 결과 ---")
    print(forecast_df[['제품코드', '예측_기간', '예측수량']].head(20))
else:
    print("❌ 예측 가능한 결과가 없습니다. (데이터 부족 또는 전처리 오류)")