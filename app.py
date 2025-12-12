import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import os
import json

# 1. 환경 설정
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

st.set_page_config(page_title="AI 분석기 Turbo", page_icon="⚡", layout="wide")

# 2. API 키 설정
api_key = None
try:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
except:
    pass

if not api_key:
    if "GOOGLE_API_KEY" in os.environ:
        api_key = os.environ["GOOGLE_API_KEY"]
    else:
        with st.sidebar:
            st.warning("⚠️ API 키 설정 필요")
            api_key = st.text_input("Google API Key 입력", type="password")

if not api_key:
    st.info("👈 왼쪽 사이드바에 API 키를 입력해주세요.")
    st.stop()

# 3. 모델 설정 (JSON 모드 활용을 위해 설정 변경)
genai.configure(api_key=api_key)
# temperature=0으로 설정하여 답변의 일관성을 높임 (정확도 향상)
generation_config = {
    "temperature": 0.0,
    "response_mime_type": "application/json",
}
model = genai.GenerativeModel("gemini-1.5-flash", generation_config=generation_config)

# 4. 배치 처리 함수 (핵심 개선 부분)
def analyze_batch(batch_df, categories):
    # 데이터를 JSON 형태로 변환하여 프롬프트에 입력
    data_str = batch_df[['id', 'comment']].to_json(orient='records', force_ascii=False)
    
    prompt = f"""
    너는 데이터 분석 전문가야. 아래 JSON 데이터의 'comment'를 읽고, 
    주제 목록: [{categories}] 중 가장 적절한 하나를 선택해서 분류해.
    
    [데이터]
    {data_str}
    
    [출력 규칙]
    반드시 아래와 같은 JSON 형식 리스트로만 출력해. 다른 말은 절대 하지 마.
    [
        {{"id": 0, "category": "선택한주제"}},
        {{"id": 1, "category": "선택한주제"}}
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text) # JSON으로 바로 파싱
    except Exception as e:
        print(f"Batch Error: {e}")
        return None

# --- 메인 UI ---
st.title("⚡ AI 데이터 분석기 (Turbo Ver.)")
st.caption("묶음 처리(Batch) 기술을 적용하여 속도와 정확도를 획기적으로 개선했습니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # 인덱스 추적을 위해 id 컬럼 임시 생성
    if 'id' not in df.columns:
        df['id'] = range(len(df))
        
    st.dataframe(df.head())

    if 'comment' not in df.columns:
        st.error("❌ 'comment' 열이 필요합니다.")
    else:
        st.markdown("---")
        st.subheader("2. 분석 모드")
        
        mode = st.radio("옵션", ["A. AI 자동 주제 탐색", "B. 고정 분류 (긍정/부정/중립/질문)"], index=1)
        
        # 주제 선정 로직
        final_cats = ""
        if "자동" in mode:
            if st.button("주제 탐색 시작"):
                with st.spinner("샘플 데이터 분석 중..."):
                    # 샘플링은 기존 방식대로 빠르게 텍스트로 처리
                    sample_txt = df['comment'].head(20).tolist()
                    temp_model = genai.GenerativeModel("gemini-1.5-flash") # 일반 텍스트 모드 모델
                    p = f"이 댓글들의 핵심 주제 4가지를 쉼표로 구분해: {sample_txt}"
                    final_cats = temp_model.generate_content(p).text.strip()
                    st.session_state.cats = final_cats
        else:
            st.session_state.cats = "긍정, 부정, 중립, 질문"

        if "cats" in st.session_state:
            st.success(f"🎯 분류 기준: {st.session_state.cats}")
            
            if st.button("🚀 고속 분석 시작"):
                results_map = {} # id: result 매핑용
                batch_size = 30  # 한 번에 30개씩 처리 (속도 조절)
                total_rows = len(df)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 배치 루프 시작
                for start_idx in range(0, total_rows, batch_size):
                    end_idx = min(start_idx + batch_size, total_rows)
                    batch_df = df.iloc[start_idx:end_idx]
                    
                    status_text.text(f"현재 {start_idx}~{end_idx}행 묶음 분석 중... ⚡")
                    
                    # API 호출 (재시도 로직 포함)
                    success = False
                    retry_count = 0
                    while not success and retry_count < 3:
                        response_data = analyze_batch(batch_df, st.session_state.cats)
                        if response_data:
                            # 결과 매핑
                            for item in response_data:
                                results_map[item['id']] = item['category']
                            success = True
                        else:
                            retry_count += 1
                            time.sleep(2) # 에러 시 대기
                    
                    if not success:
                        st.warning(f"{start_idx}번 구간 분석 실패. 건너뜁니다.")

                    # 진행률 업데이트
                    progress_bar.progress(end_idx / total_rows)
                    time.sleep(1) # API 과부하 방지용 짧은 대기 (배치 간격)

                # 결과 병합
                df['분석_결과'] = df['id'].map(results_map)
                
                st.write("### ✅ 최종 결과")
                # id 열은 제거하고 보여주기
                st.dataframe(df.drop(columns=['id']))
                
                csv = df.drop(columns=['id']).to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 결과 다운로드", csv, "result.csv", "text/csv")
