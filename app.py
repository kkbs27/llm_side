import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import pandas as pd
import time
import os

# 1. 시스템 설정
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
st.set_page_config(page_title="AI 분석기 (Tank Ver.)", page_icon="🛡️", layout="wide")

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
            st.warning("⚠️ API 키 필요")
            api_key = st.text_input("API Key 입력", type="password")

if not api_key:
    st.stop()

# 3. 모델 설정 (안전장치 완전 해제)
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# [핵심] 욕설, 비하, 선정성 등 모든 필터 해제 (BLOCK_NONE)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# 4. 분석 함수 (하나씩 처리)
def analyze_one_comment(comment, categories):
    prompt = f"""
    다음 댓글을 [{categories}] 중 하나로만 분류해.
    
    [댓글] {comment}
    
    [조건]
    1. 설명하지 마.
    2. 오직 단어 하나만 출력해.
    """
    try:
        # 안전 설정 적용하여 호출
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text.strip()
    except Exception as e:
        # 에러 발생 시 로그 출력
        print(f"Error for '{comment}': {e}")
        if "429" in str(e):
            return "RATE_LIMIT"
        return "ERROR"

# --- 메인 UI ---
st.title("🛡️ AI 분석기 (안전모드)")
st.caption("속도는 조금 느리지만, 욕설/비판 데이터도 한 줄씩 정확하게 분석합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    # 인코딩 자동 감지
    encodings = ['utf-8', 'cp949', 'euc-kr']
    df = None
    for code in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=code, engine='python', on_bad_lines='skip')
            break 
        except:
            continue
            
    if df is None:
        st.error("파일 읽기 실패")
        st.stop()

    if 'comment' not in df.columns:
        st.error("'comment' 열이 없습니다.")
    else:
        st.dataframe(df.head())
        
        st.markdown("---")
        # 간단하게 버튼 하나로 통일
        if st.button("분석 시작 (긍정/부정/중립/질문) 🚀"):
            
            cats = "긍정, 부정, 중립, 질문"
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            total_rows = len(df)
            
            # [핵심] 한 줄씩 또박또박 처리
            for i, row in df.iterrows():
                comment = row['comment']
                
                # 분석 실행
                res = analyze_one_comment(comment, cats)
                
                # 속도 제한(429) 걸리면 5초 쉬고 재시도
                if res == "RATE_LIMIT":
                    time.sleep(5)
                    res = analyze_one_comment(comment, cats) # 재시도
                
                results.append(res)
                
                # 1초 대기 (안정성 확보)
                time.sleep(1)
                
                # 진행상황
                progress_bar.progress((i + 1) / total_rows)
                status_text.text(f"분석 중... ({i + 1}/{total_rows}) : {res}") # 현재 결과 보여줌
                
            df['분석_결과'] = results
            st.success("완료!")
            st.dataframe(df)
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("결과 다운로드", csv, "final_result.csv", "text/csv")
