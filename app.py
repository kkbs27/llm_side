import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold # 안전 설정용
import pandas as pd
import time
import os

# 1. 경고 메시지 차단
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 2. 페이지 설정
st.set_page_config(page_title="AI 분석기 Final (Uncensored)", page_icon="⚡", layout="wide")

# 3. API 키 설정
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

# 4. 모델 설정 (안전 필터 해제 설정 추가)
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# [핵심] 욕설/비하 발언도 분석할 수 있게 안전장치를 끔 (BLOCK_NONE)
safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# 5. 재시도 함수
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            # 안전 설정 적용하여 호출
            response = model.generate_content(prompt, safety_settings=safety_settings)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota" in error_msg:
                with st.sidebar:
                    st.toast(f"🚦 잠시 대기 중... ({attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                time.sleep(1)
    return "FAIL"

# --- 메인 UI ---
st.title("⚡ AI 데이터 분석기 (No Filter Ver.)")
st.caption("안전 필터를 해제하여 욕설/비판 댓글도 정확히 분석합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
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
        st.error("❌ 파일을 읽을 수 없습니다.")
        st.stop()

    st.write("### 1. 데이터 확인")
    st.dataframe(df.head())

    if 'comment' not in df.columns:
        st.error("❌ 'comment' 열이 없습니다.")
    else:
        st.markdown("---")
        st.subheader("2. 분석 모드 선택")
        
        analysis_mode = st.radio(
            "어떤 기준으로 분류할까요?",
            ["A. AI가 주제 스스로 찾기 (고급)", "B. 긍정/부정/중립/질문 (기본)"],
            index=1
        )

        if "스스로" in analysis_mode:
            if st.button("Step 1. 주제 탐색 시작"):
                with st.spinner("주제 분석 중..."):
                    sample = df['comment'].head(20).tolist()
                    prompt = f"다음 댓글들을 읽고 핵심 주제 4가지를 쉼표로 구분해줘. 예: 맛,가격,배송,서비스 \n\n[댓글]: {sample}"
                    categories = generate_with_retry(prompt)
                    st.session_state.final_categories = categories
                    st.success(f"✅ 발견된 주제: {categories}")
        else:
            if st.button("Step 1. 기준 설정"):
                st.session_state.final_categories = "긍정, 부정, 중립, 질문"
                st.success("✅ 기준 설정됨: 긍정, 부정, 중립, 질문")

        # --- 고속 배치 처리 로직 ---
        if "final_categories" in st.session_state:
            st.markdown("---")
            st.write(f"### 🎯 기준: **[{st.session_state.final_categories}]**")
            
            if st.button("Step 2. 고속 분류 시작 (Batch) 🚀"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                BATCH_SIZE = 10 
                target_df = df
                total_rows = len(target_df)
                cats = st.session_state.final_categories

                for i in range(0, total_rows, BATCH_SIZE):
                    batch = target_df.iloc[i : i + BATCH_SIZE]
                    batch_comments = batch['comment'].tolist()
                    
                    # [핵심 수정] 리스트 말고 그냥 파이프(|)로 나누라고 지시 (훨씬 잘 알아들음)
                    prompt = f"""
                    다음 {len(batch_comments)}개의 댓글을 [{cats}] 중 하나로 분류해.
                    
                    [댓글 목록]
                    {batch_comments}
                    
                    [조건]
                    1. 결과는 반드시 수직선(|) 기호로 구분해서 한 줄로 출력해.
                    2. 예시: 긍정|부정|중립
                    3. 다른 말 하지 말고 오직 결과만 줘.
                    4. 개수는 정확히 {len(batch_comments)}개여야 해.
                    """
                    
                    res_text = generate_with_retry(prompt)
                    
                    # [파싱 로직 단순화] 그냥 | 로 자름
                    try:
                        # 혹시 모를 마크다운 제거
                        clean_text = res_text.replace("```", "").strip()
                        batch_results = clean_text.split("|")
                        
                        # 공백 제거
                        batch_results = [r.strip() for r in batch_results]

                        if len(batch_results) != len(batch_comments):
                            # 개수 안 맞으면 에러 로그 출력해봄
                            print(f"개수 불일치! 기대: {len(batch_comments)}, 실제: {len(batch_results)}")
                            print(f"AI 응답: {clean_text}")
                            # 부족하면 채우기
                            if len(batch_results) < len(batch_comments):
                                batch_results.extend(["판독불가"] * (len(batch_comments) - len(batch_results)))
                            else:
                                batch_results = batch_results[:len(batch_comments)]
                            
                    except Exception as e:
                        batch_results = ["판독불가"] * len(batch_comments)
                    
                    results.extend(batch_results)
                    time.sleep(1) # 1초 대기
                    
                    current_progress = min((i + BATCH_SIZE) / total_rows, 1.0)
                    progress_bar.progress(current_progress)
                    status_text.text(f"🚀 고속 분석 중... ({min(i + BATCH_SIZE, total_rows)}/{total_rows})")

                if len(results) < total_rows:
                    results.extend(["미처리"] * (total_rows - len(results)))
                elif len(results) > total_rows:
                    results = results[:total_rows]

                df['분석_결과'] = results
                status_text.text("✅ 분석 완료!")
                st.success("분석 완료!")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("결과 CSV 다운로드", csv, "fast_result.csv", "text/csv")
