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
# 배포 시엔 Secrets에서, 로컬에선 입력창에서 받음
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

# 4. 모델 설정 (사용자 환경에 최적화된 2.0 Flash 사용)
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# 5. 재시도 함수 (에러 방지용)
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}") # 로그 출력
            
            if "429" in error_msg or "Quota" in error_msg:
                # 속도 제한 걸리면 조금 길게 대기
                with st.sidebar:
                    st.toast(f"🚦 속도 조절 중... ({attempt+1}/{max_retries})")
                time.sleep(10)
            else:
                time.sleep(1)
    return "FAIL"

# --- 메인 UI 시작 ---
st.title("📊 AI 데이터 분석기 (Pro Ver.)")
st.caption("AI가 데이터를 먼저 읽고 주제를 찾거나, 정해진 기준으로 분류합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드 (열 이름 'comment' 필수)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. 데이터 확인")
    st.dataframe(df.head())

    if 'comment' not in df.columns:
        st.error("❌ CSV 파일에 'comment' 열이 없습니다. 확인해주세요.")
    else:
        st.markdown("---")
        st.subheader("2. 분석 모드 선택")
        
        # [핵심 기능] 모드 선택 라디오 버튼
        analysis_mode = st.radio(
            "어떤 기준으로 분류할까요?",
            ["A. AI가 주제 스스로 찾기 (고급)", "B. 긍정/부정/중립/질문 (기본)"],
            index=1 # 기본값을 B로 두어 안정성 확보
        )

        # 모드에 따른 로직 분기
        if "스스로" in analysis_mode:
            st.info("🕵️ AI가 데이터 일부를 먼저 읽고, 가장 중요한 주제 4가지를 뽑아냅니다.")
            if st.button("Step 1. 주제 탐색 시작"):
                with st.spinner("데이터를 분석하여 분류 기준을 수립 중입니다..."):
                    # 샘플링
                    sample_comments = df['comment'].head(20).tolist()
                    
                    discovery_prompt = f"""
                    너는 데이터 분석가야. 아래 댓글들을 읽고 전체를 관통하는 핵심 주제를 딱 4가지로 요약해.
                    [댓글 샘플] {sample_comments}
                    [조건]
                    1. 주제 4개는 쉼표(,)로만 구분해. (예: 가격, 품질, 배송, 서비스)
                    2. 설명이나 번호 매기기 절대 금지. 오직 단어 4개만 출력해.
                    """
                    
                    categories = generate_with_retry(discovery_prompt)
                    
                    if "FAIL" in categories:
                        st.error("주제 발견에 실패했습니다. 잠시 후 다시 시도해주세요.")
                    else:
                        st.session_state.final_categories = categories
                        st.success(f"✅ AI가 발견한 주제: {categories}")
        
        else: # B 모드 (기본)
            st.info("💡 가장 보편적인 [긍정, 부정, 중립, 질문] 4가지 기준으로 분류합니다.")
            if st.button("Step 1. 기준 설정"):
                st.session_state.final_categories = "긍정, 부정, 중립, 질문"
                st.success("✅ 분류 기준 설정됨: 긍정, 부정, 중립, 질문")

        # 3. 전체 분류 실행 (공통 로직)
        if "final_categories" in st.session_state:
            st.markdown("---")
            st.write(f"### 🎯 확정된 기준: **[{st.session_state.final_categories}]**")
            
            if st.button("Step 2. 전체 데이터 분류 시작 🚀"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                results = []
                # 전체 데이터 대상
                target_df = df
                total_rows = len(target_df)
                cats = st.session_state.final_categories

                for i, row in target_df.iterrows():
                    comment = row['comment']
                    
                    prompt = f"""
                    다음 댓글을 [{cats}] 중 하나로만 분류해.
                    [댓글] {comment}
                    [조건] 설명 없이 딱 단어 하나만 출력해.
                    """
                    
                    res = generate_with_retry(prompt)
                    results.append(res)
                    
                    # [필수] 속도 제한 방지를 위한 2초 대기
                    time.sleep(2)
                    
                    # 진행상황 업데이트
                    progress_bar.progress((i + 1) / total_rows)
                    status_text.text(f"AI가 열심히 분석 중... ({i + 1}/{total_rows})")
                
                # 결과 저장
                df['분석_결과'] = results
                status_text.text("✅ 모든 분석이 완료되었습니다!")
                st.success("분석 완료!")
                
                st.write("### 최종 결과")
                st.dataframe(df)
                
                # 다운로드 버튼
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 결과 CSV 다운로드",
                    data=csv,
                    file_name="ai_analysis_result.csv",
                    mime="text/csv"
                )
