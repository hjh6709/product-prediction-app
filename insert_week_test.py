import pandas as pd
from sqlalchemy import create_engine
import urllib
from tqdm import tqdm
import lightgbm as lgb
import numpy as np

# ✅ 1. DB 연결 설정
server = 'localhost'
database = 'SPTEST1'
driver = 'ODBC Driver 17 for SQL Server'
params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
)
engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

# ✅ 2. 데이터 불러오기
try:
    # 일자별 데이터가 저장된 'T출고' 테이블에서 데이터 로드
    df = pd.read_sql("SELECT * FROM T출고", engine)
    print("--- 'T출고' 테이블에서 데이터 로드 성공 ---")
except Exception as e:
    print(f"--- 'T출고' 테이블에서 데이터 로드 실패: {e} ---")
    print("DB 연결 문자열, 서버 상태, 데이터베이스/테이블 이름, ODBC 드라이버 설치를 확인하세요.")
    exit()

# --- DEBUG: 원본 df 정보 확인 ---
print("\n--- DEBUG: DB에서 불러온 원본 df 정보 ---")
print(f"DataFrame이 비어있는가?: {df.empty}")
print(f"총 행 수: {len(df)}")
print(df.info()) # 각 컬럼의 데이터 타입 및 결측치 확인
print(df.head()) # 상위 5행 데이터 확인

if '출고일자' not in df.columns:
    print("\n🚨🚨 경고: '출고일자' 컬럼이 df에 존재하지 않습니다! 컬럼명을 확인해주세요. 🚨🚨")
    exit()
# -----------------------------------------------------------

# ✅ 3. 날짜 처리 및 주차, 시간 기반 피처 추출
# '출고일자' 컬럼을 datetime 형식으로 변환 (이미 DB에 넣을 때 변환되었지만, 다시 확인)
df['출고일자'] = pd.to_datetime(df['출고일자'], errors='coerce')
df = df.dropna(subset=['출고일자']) # 날짜 변환 실패한 행 제거

if len(df) == 0:
    print("\n❌ '출고일자' 변환 후 모든 데이터가 제거되었습니다. '출고일자' 컬럼의 형식을 확인하세요. ❌")
    exit()

# --- ⭐ 새로운 피처: ISO 캘린더 기반 년도, 주차, 요일 추출 및 기타 시간 피처 ⭐ ---
# '출고년도'와 '출고주차'는 .isocalendar() 결과 DataFrame에서 추출해야 합니다.
# '출고요일'은 .dt.weekday + 1을 사용합니다.
df['출고년도'] = df['출고일자'].dt.isocalendar().year.astype(int)
df['출고주차'] = df['출고일자'].dt.isocalendar().week.astype(int)
# ⭐⭐⭐ 수정된 부분: df['출고요일'] 정의 시 .dt.weekday + 1 사용 ⭐⭐⭐
df['출고요일'] = (df['출고일자'].dt.weekday + 1).astype(int) # 월=1, 일=7 (ISO 8601 표준)

df['월'] = df['출고일자'].dt.month
df['분기'] = df['출고일자'].dt.quarter
df['일'] = df['출고일자'].dt.day 
# df['주차_시작일']은 출고요일을 기반으로 하므로, 출고요일 정의 후 계산
df['주차_시작일'] = df['출고일자'] - pd.to_timedelta(df['출고요일'] - 1, unit='D') # 해당 주의 월요일 날짜
# ----------------------------------------------------------------------------------

print("\n--- '출고일자' 처리 및 피처 추출 후 df 샘플 ---")
print(df[['출고일자', '출고년도', '출고주차', '출고요일', '월', '분기', '주차_시작일']].head())


# ✅ 4. 제품코드 + 년도 + 주차 단위 집계
weekly_df = df.groupby(['제품코드', '출고년도', '출고주차', '주차_시작일', '월', '분기', '출고요일'])['확정수량'].sum().reset_index()
weekly_df = weekly_df.sort_values(by=['제품코드', '출고년도', '출고주차']).copy()
weekly_df['주차순서'] = weekly_df.groupby('제품코드').cumcount() # 각 제품별 주차의 순서

# --- ⭐ 새로운 피처: 과거 값 (Lag Features) 및 이동 평균 추가 (주차 단위) ⭐ ---
weekly_df['확정수량_직전주'] = weekly_df.groupby('제품코드')['확정수량'].shift(1)
weekly_df['확정수량_2주전'] = weekly_df.groupby('제품코드')['확정수량'].shift(2)
weekly_df['확정수량_1년전'] = weekly_df.groupby('제품코드')['확정수량'].shift(52) # 1년은 약 52주
weekly_df['확정수량_4주평균'] = weekly_df.groupby('제품코드')['확정수량'].rolling(window=4, min_periods=1).mean().reset_index(level=0, drop=True)

# 새로운 피처들로 인해 NaN이 생길 수 있습니다. 모델 학습 전에 이들을 처리합니다.
weekly_df.fillna(0, inplace=True) # NaN 값을 0으로 채우기 (간단한 처리)

print("\n--- weekly_df head (새로운 피처 포함) ---")
print(weekly_df.head())
print(f"주차별 집계 후 총 데이터 행 수: {len(weekly_df)}")

if len(weekly_df) == 0:
    print("\n❌ 주차별 집계 후 데이터가 없습니다. 제품코드 또는 확정수량에 문제가 있을 수 있습니다. ❌")
    exit()

print("--- 각 제품코드별 주차별 데이터 개수 (상위 10개) ---")
product_counts = weekly_df['제품코드'].value_counts()
print(product_counts.head(10))


# ✅ 5. 예측 (LightGBM 사용)
results = []
codes = weekly_df['제품코드'].unique()
print(f"\n총 제품 수: {len(codes)}")

if len(codes) == 0:
    print("❌ 예측할 제품 코드가 없습니다.")
else:
    예측_시작_날짜 = pd.to_datetime('2025-08-04')
    예측_종료_날짜 = pd.to_datetime('2025-10-27')

    predict_dates_range = []
    current_date = 예측_시작_날짜
    while current_date <= 예측_종료_날짜:
        predict_dates_range.append(current_date)
        current_date += pd.Timedelta(weeks=1)

    num_forecast_weeks = len(predict_dates_range)

    features = [
        '주차순서', '출고년도', '출고주차', '월', '분기', '출고요일', 
        '확정수량_직전주', '확정수량_2주전', '확정수량_1년전', '확정수량_4주평균'
    ]


    for code in tqdm(codes, desc="주차별 예측 중"):
        sub = weekly_df[weekly_df['제품코드'] == code].copy()

        if len(sub) < 60:
            continue

        X_train = sub[features]
        y_train = sub['확정수량']

        if X_train.isnull().all().all():
             continue


        model = lgb.LGBMRegressor(random_state=42)
        try:
            model.fit(X_train, y_train)
        except lgb.basic.LightGBMError as e:
            continue


        last_observed_row = sub.iloc[-1]
        
        predicted_lag1 = last_observed_row['확정수량']
        predicted_lag2 = last_observed_row['확정수량_직전주'] 
        
        def get_lag_year_value(product_code, year, week, weekly_df_data):
            prev_year_data = weekly_df_data[
                (weekly_df_data['제품코드'] == product_code) &
                (weekly_df_data['출고년도'] == year - 1) &
                (weekly_df_data['출고주차'] == week)      
            ]
            if not prev_year_data.empty:
                return prev_year_data['확정수량'].iloc[0]
            return 0 

        recent_quantities_for_rolling = sub['확정수량'].tail(4).tolist()

        current_product_forecast_list = []

        for i in range(num_forecast_weeks):
            current_date_for_prediction = predict_dates_range[i]
            
            last_train_week_date = sub['주차_시작일'].max()
            last_train_week_order = sub[sub['주차_시작일'] == last_train_week_date]['주차순서'].iloc[0]

            time_diff_in_weeks = (current_date_for_prediction - last_train_week_date).days // 7
            future_order_for_model = last_train_week_order + time_diff_in_weeks


            lag_year_value = get_lag_year_value(code, current_date_for_prediction.year, current_date_for_prediction.isocalendar().week, weekly_df)
            
            current_rolling_mean = np.mean(recent_quantities_for_rolling[-4:]) if len(recent_quantities_for_rolling) >= 1 else 0


            future_features_dict = {
                '주차순서': future_order_for_model,
                '출고년도': current_date_for_prediction.year,
                '출고주차': current_date_for_prediction.isocalendar().week,
                '월': current_date_for_prediction.month,
                '분기': current_date_for_prediction.quarter,
                # ⭐⭐⭐ 수정된 부분: .weekday() + 1 사용 ⭐⭐⭐
                '출고요일': current_date_for_prediction.weekday() + 1, 
                '확정수량_직전주': predicted_lag1,
                '확정수량_2주전': predicted_lag2,
                '확정수량_1년전': lag_year_value,
                '확정수량_4주평균': current_rolling_mean
            }
            
            future_df_for_prediction = pd.DataFrame([future_features_dict], columns=features)
            
            predicted_quantity = model.predict(future_df_for_prediction)[0]
            predicted_quantity = max(0, int(round(predicted_quantity)))

            predicted_lag2 = predicted_lag1
            predicted_lag1 = predicted_quantity
            
            recent_quantities_for_rolling.append(predicted_quantity)
            if len(recent_quantities_for_rolling) > 4:
                recent_quantities_for_rolling.pop(0)

            year_str = current_date_for_prediction.strftime('%Y년')
            month_str = current_date_for_prediction.strftime('%m월')
            
            first_day_of_month = current_date_for_prediction.replace(day=1)
            first_iso_week_of_month = first_day_of_month.isocalendar().week
            
            current_iso_week = current_date_for_prediction.isocalendar().week
            week_of_month = current_iso_week - first_iso_week_of_month + 1
            if week_of_month <= 0:
                week_of_month = 1
            
            week_str = f'{int(week_of_month)}주차'

            current_product_forecast_list.append({
                '제품코드': code,
                '예측_기간': f'{year_str} {month_str} {week_str}',
                '예측수량': predicted_quantity
            })
        
        results.extend(current_product_forecast_list)

# ✅ 6. 결과 출력
if results:
    forecast_df = pd.DataFrame(results)
    forecast_df['예측_기간_월_시작일'] = forecast_df['예측_기간'].apply(lambda x: pd.to_datetime(x.split(' ')[0] + x.split(' ')[1], format='%Y년%m월'))
    forecast_df['예측_기간_주차번호'] = forecast_df['예측_기간'].apply(lambda x: int(x.split(' ')[2].replace('주차', '')))
    
    forecast_df = forecast_df.sort_values(by=['제품코드', '예측_기간_월_시작일', '예측_기간_주차번호']).drop(columns=['예측_기간_월_시작일', '예측_기간_주차번호'])

    print("\n--- 최종 예측 결과 ---")
    print(forecast_df)
else:
    print("❌ 예측 가능한 결과가 없습니다. (데이터 부족 또는 전처리 오류)")