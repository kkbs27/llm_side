import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import os

# 1. 보기 싫은 경고 메시지 차단
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 2. 페이지 설정
st.set_page_config(page_title="AI 분석기 Final", page_icon="📊", layout="wide")

# 3. API 키 설정 (하이브리드 방식)
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

genai.configure(api_key=api_key)

# [수정됨] 사용자 목록에 있는 'gemini-2.0-flash' 사용
model = genai.GenerativeModel("gemini-2.0-flash")

# --- 재시도 함수 ---
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}") # 터미널에 에러 로그 출력
            
            # 429(속도제한)나 Quota 에러 시
            if "429" in error_msg or "Quota" in error_msg:
                with st.sidebar:
                    st.toast(f"🚦 속도 조절 중... 잠시 대기 ({attempt+1}/{max_retries})")
                time.sleep(10) # 10초 대기
            else:
                time.sleep(1) # 일반 에러는 1초 대기
                
    return "FAIL"

# --- 메인 UI ---
st.title("📊 AI 데이터 분석기 (Final Ver.)")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. 데이터 확인")
    st.dataframe(df.head())

    if 'comment' not in df.columns:
        st.error("❌ 'comment' 열이 없습니다.")
    else:
        st.markdown("---")
        st.subheader("2. 분석 실행")
        
        if st.button("분석 시작 (긍정/부정/중립/질문)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            # 전체 데이터 분석
            target_df = df 
            total_rows = len(target_df)

            for i, row in target_df.iterrows():
                comment = row['comment']
                prompt = f"다음 댓글을 [긍정, 부정, 중립, 질문] 중 하나로만 분류해. 댓글: {comment}"
                
                res = generate_with_retry(prompt)
                results.append(res)
                
                # [핵심] 2초씩 무조건 쉬기 (과속 단속 회피)
                time.sleep(2) 
                
                # 진행률 업데이트
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"분석 중... ({i + 1}/{total_rows})")
            
            # 결과 저장
            df['분석_결과'] = results
            
            st.success("완료!")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("결과 다운로드", csv, "result.csv", "text/csv")
