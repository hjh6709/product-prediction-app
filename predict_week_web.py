import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import urllib
from tqdm import tqdm
import lightgbm as lgb
import numpy as np

# Streamlit 앱의 제목 설정
st.set_page_config(page_title="제품 출고량 주차별 예측", layout="wide")
st.title("📦 제품 출고량 주차별 예측 시스템")

st.markdown("""
이 앱은 제품별 주차별 출고량을 예측합니다.
데이터베이스에서 과거 출고 데이터를 불러와 모델을 학습하고,
지정된 기간 동안의 미래 출고량을 예측합니다.
""")

# --- DB 연결 설정 (기존 코드와 동일) ---
server = 'localhost'
database = 'SPTEST1'
driver = 'ODBC Driver 17 for SQL Server'
params = urllib.parse.quote_plus(
    f"DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;"
)

# SQLAlchemy 엔진은 한 번만 생성하는 것이 효율적입니다.
@st.cache_resource
def get_db_engine():
    """데이터베이스 엔진을 캐싱하여 여러 번 생성하지 않도록 합니다."""
    try:
        engine = create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
        # 연결 테스트 (선택 사항)
        with engine.connect() as connection:
            connection.execute(pd.text('SELECT 1'))
        st.success("데이터베이스 연결 성공! ✅")
        return engine
    except Exception as e:
        st.error(f"데이터베이스 연결 실패: {e} ❌")
        st.warning("DB 연결 문자열, 서버 상태, 데이터베이스/테이블 이름, ODBC 드라이버 설치를 확인하세요.")
        return None

engine = get_db_engine()

# --- 데이터 불러오기 및 전처리 (기존 코드와 동일) ---
@st.cache_data(show_spinner="데이터를 불러오고 전처리 중...") # 데이터를 캐싱하여 앱 재실행 시 다시 로드하지 않도록 함
def load_and_preprocess_data(engine):
    if engine is None:
        return pd.DataFrame(), pd.DataFrame(), "DB 연결 실패"

    try:
        df = pd.read_sql("SELECT * FROM T출고", engine)
        
        if '출고일자' not in df.columns:
            return pd.DataFrame(), pd.DataFrame(), "'출고일자' 컬럼이 데이터에 없습니다."

        df['출고일자'] = pd.to_datetime(df['출고일자'], errors='coerce')
        df = df.dropna(subset=['출고일자'])

        if len(df) == 0:
            return pd.DataFrame(), pd.DataFrame(), "'출고일자' 변환 후 모든 데이터가 제거되었습니다."

        df['출고년도'] = df['출고일자'].dt.isocalendar().year.astype(int)
        df['출고주차'] = df['출고일자'].dt.isocalendar().week.astype(int)
        df['출고요일'] = (df['출고일자'].dt.weekday + 1).astype(int)
        df['월'] = df['출고일자'].dt.month
        df['분기'] = df['출고일자'].dt.quarter
        df['일'] = df['출고일자'].dt.day 
        df['주차_시작일'] = df['출고일자'] - pd.to_timedelta(df['출고요일'] - 1, unit='D')

        weekly_df = df.groupby(['제품코드', '출고년도', '출고주차', '주차_시작일', '월', '분기', '출고요일'])['확정수량'].sum().reset_index()
        weekly_df = weekly_df.sort_values(by=['제품코드', '출고년도', '출고주차']).copy()
        weekly_df['주차순서'] = weekly_df.groupby('제품코드').cumcount()

        weekly_df['확정수량_직전주'] = weekly_df.groupby('제품코드')['확정수량'].shift(1)
        weekly_df['확정수량_2주전'] = weekly_df.groupby('제품코드')['확정수량'].shift(2)
        weekly_df['확정수량_1년전'] = weekly_df.groupby('제품코드')['확정수량'].shift(52)
        weekly_df['확정수량_4주평균'] = weekly_df.groupby('제품코드')['확정수량'].rolling(window=4, min_periods=1).mean().reset_index(level=0, drop=True)
        weekly_df.fillna(0, inplace=True)

        if len(weekly_df) == 0:
            return pd.DataFrame(), pd.DataFrame(), "주차별 집계 후 데이터가 없습니다."
        
        st.success(f"데이터 로드 및 전처리 완료! 총 {len(weekly_df)} 주차별 데이터 로드.")
        return df, weekly_df, None

    except Exception as e:
        st.error(f"데이터 로드 및 전처리 중 오류 발생: {e} ❌")
        return pd.DataFrame(), pd.DataFrame(), "데이터 처리 오류"

# 데이터 로드 및 전처리
df_raw, weekly_df, data_load_error = load_and_preprocess_data(engine)

if data_load_error:
    st.error(data_load_error)
    st.stop() # 오류 발생 시 앱 실행 중지

# --- 예측 로직 (기존 코드와 동일) ---
@st.cache_data(show_spinner="예측 모델 학습 및 예측 수행 중...")
def run_prediction(weekly_df_data, start_date, end_date):
    results = []
    codes = weekly_df_data['제품코드'].unique()
    
    st.info(f"총 {len(codes)}개의 제품에 대해 예측을 시작합니다.")

    predict_dates_range = []
    current_date = start_date
    while current_date <= end_date:
        predict_dates_range.append(current_date)
        current_date += pd.Timedelta(weeks=1)

    num_forecast_weeks = len(predict_dates_range)

    features = [
        '주차순서', '출고년도', '출고주차', '월', '분기', '출고요일', 
        '확정수량_직전주', '확정수량_2주전', '확정수량_1년전', '확정수량_4주평균'
    ]

    progress_bar = st.progress(0)
    for idx, code in enumerate(tqdm(codes, desc="제품별 예측 진행 중")):
        sub = weekly_df_data[weekly_df_data['제품코드'] == code].copy()

        if len(sub) < 60:
            # st.warning(f"제품코드 {code}: 학습 데이터가 60주 미만이므로 예측에서 제외됩니다.")
            continue

        X_train = sub[features]
        y_train = sub['확정수량']

        if X_train.isnull().all().all():
            # st.warning(f"제품코드 {code}: 훈련 데이터가 모두 Null이므로 예측에서 제외됩니다.")
            continue

        model = lgb.LGBMRegressor(random_state=42, verbose=-1) # verbose=-1 로 경고 메시지 숨김
        try:
            model.fit(X_train, y_train)
        except lgb.basic.LightGBMError as e:
            # st.warning(f"제품코드 {code}: 모델 학습 중 오류 발생({e}). 예측에서 제외됩니다.")
            continue

        last_observed_row = sub.iloc[-1]
        
        predicted_lag1 = last_observed_row['확정수량']
        predicted_lag2 = last_observed_row['확정수량_직전주'] 
        
        def get_lag_year_value(product_code, year, week, weekly_df_data_inner):
            prev_year_data = weekly_df_data_inner[
                (weekly_df_data_inner['제품코드'] == product_code) &
                (weekly_df_data_inner['출고년도'] == year - 1) &
                (weekly_df_data_inner['출고주차'] == week)      
            ]
            if not prev_year_data.empty:
                return prev_year_data['확정수량'].iloc[0]
            return 0 

        recent_quantities_for_rolling = sub['확정수량'].tail(4).tolist()

        current_product_forecast_list = []

        for j in range(num_forecast_weeks):
            current_date_for_prediction = predict_dates_range[j]
            
            last_train_week_date = sub['주차_시작일'].max()
            # 마지막 훈련 데이터의 주차순서가 예측 시작 날짜보다 미래일 경우를 방지
            # 예측 시작 날짜가 훈련 데이터의 마지막 날짜보다 항상 미래여야 함
            if last_train_week_date is pd.NaT or current_date_for_prediction < last_train_week_date:
                # 이 경우는 예측할 데이터가 아니므로 스킵하거나 적절히 처리
                # 여기서는 단순히 스킵합니다.
                continue 

            last_train_week_order = sub[sub['주차_시작일'] == last_train_week_date]['주차순서'].iloc[0]

            time_diff_in_weeks = (current_date_for_prediction - last_train_week_date).days // 7
            future_order_for_model = last_train_week_order + time_diff_in_weeks


            lag_year_value = get_lag_year_value(code, current_date_for_prediction.year, current_date_for_prediction.isocalendar().week, weekly_df_data)
            
            current_rolling_mean = np.mean(recent_quantities_for_rolling[-4:]) if len(recent_quantities_for_rolling) >= 1 else 0


            future_features_dict = {
                '주차순서': future_order_for_model,
                '출고년도': current_date_for_prediction.year,
                '출고주차': current_date_for_prediction.isocalendar().week,
                '월': current_date_for_prediction.month,
                '분기': current_date_for_prediction.quarter,
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
        progress_bar.progress((idx + 1) / len(codes))
    
    return results

# --- Streamlit UI 구성 ---
st.header("예측 설정")

# 예측 기간 입력
col1, col2 = st.columns(2)
with col1:
    default_start_date = pd.to_datetime('2025-08-04').date() # 오늘 날짜를 기준으로 변경 (현재 2025년 7월 16일이므로 미래 날짜로 설정)
    st.date_input("예측 시작 날짜를 선택하세요:", value=default_start_date, key="start_date")
with col2:
    default_end_date = pd.to_datetime('2025-10-27').date() # 예측 기간을 조절
    st.date_input("예측 종료 날짜를 선택하세요:", value=default_end_date, key="end_date")

start_date_input = st.session_state.start_date
end_date_input = st.session_state.end_date

# 제품 선택 (옵션)
# product_codes = ['전체'] + sorted(weekly_df['제품코드'].unique().tolist())
# selected_product_code = st.selectbox("특정 제품 코드를 선택하세요 (선택 사항):", product_codes)


if st.button("예측 실행"):
    if engine is None:
        st.error("데이터베이스 연결이 설정되지 않아 예측을 실행할 수 없습니다.")
    elif weekly_df.empty:
        st.error("데이터를 불러오거나 전처리하는 중에 문제가 발생했습니다. 예측을 실행할 수 없습니다.")
    elif start_date_input >= end_date_input:
        st.error("예측 시작 날짜는 예측 종료 날짜보다 빨라야 합니다.")
    else:
        st.subheader("예측 결과")
        # 실제 예측 함수 호출
        # 선택된 제품 코드 필터링
        # if selected_product_code == '전체':
        #     filtered_weekly_df = weekly_df
        # else:
        #     filtered_weekly_df = weekly_df[weekly_df['제품코드'] == selected_product_code]
        
        # if filtered_weekly_df.empty:
        #     st.warning("선택된 제품 코드에 대한 데이터가 부족하여 예측을 수행할 수 없습니다.")
        # else:
        forecast_results = run_prediction(weekly_df, 
                                          pd.to_datetime(start_date_input), 
                                          pd.to_datetime(end_date_input))

        if forecast_results:
            forecast_df = pd.DataFrame(forecast_results)
            forecast_df['예측_기간_월_시작일'] = forecast_df['예측_기간'].apply(lambda x: pd.to_datetime(x.split(' ')[0] + x.split(' ')[1], format='%Y년%m월'))
            forecast_df['예측_기간_주차번호'] = forecast_df['예측_기간'].apply(lambda x: int(x.split(' ')[2].replace('주차', '')))
            
            final_forecast_df = forecast_df.sort_values(by=['제품코드', '예측_기간_월_시작일', '예측_기간_주차번호']).drop(columns=['예측_기간_월_시작일', '예측_기간_주차번호'])
            
            st.dataframe(final_forecast_df, use_container_width=True) # 데이터프레임을 웹에 표시
            
            # 예측 결과를 CSV로 다운로드할 수 있도록 버튼 추가
            csv = final_forecast_df.to_csv(index=False).encode('utf-8-sig') # 한글 깨짐 방지를 위해 utf-8-sig
            st.download_button(
                label="예측 결과 CSV 다운로드",
                data=csv,
                file_name="predicted_shipment_weekly.csv",
                mime="text/csv",
            )
        else:
            st.warning("예측 가능한 결과가 없습니다. (데이터 부족 또는 전처리 오류로 인해 예측이 수행되지 않았을 수 있습니다.)")

st.markdown("---")
st.markdown("Made with ❤️ by Your Name/Company Name")