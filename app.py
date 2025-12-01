import streamlit as st
import google.generativeai as genai
import pandas as pd
import time

# 1. 페이지 및 API 설정
st.set_page_config(page_title="AI 스마트 군집화", page_icon="🧠")

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
except:
    st.error("API 키 설정이 필요합니다.")
    st.stop()

# --- [핵심] 똑똑한 재시도 함수 정의 ---
def generate_with_retry(prompt, max_retries=3):
    """
    API 호출 실패 시 잠시 대기했다가 재시도하는 함수
    (Exponential Backoff 적용: 2초 -> 4초 -> 8초 대기)
    """
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            # 에러 발생 시 처리
            wait_time = 2 ** (attempt + 1) # 2의 n승으로 대기 시간 증가
            
            # 스트림릿 화면에 작은 경고 표시 (Toast)
            st.toast(f"API 호출량이 많아 {wait_time}초 대기 후 재시도합니다... ({attempt+1}/{max_retries})")
            time.sleep(wait_time)
            
    return "API_ERROR" # 3번 다 실패하면 에러 반환

# ------------------------------------

st.title("🧠 AI 자율 군집화 봇 (Pro)")
st.caption("데이터 분석 및 Rate Limit 자동 대응 기능 탑재")

# 2. 파일 업로드
uploaded_file = st.file_uploader("댓글 CSV 파일 업로드 ('comment' 열 필수)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 1. 데이터 확인")
    st.dataframe(df.head())

    if 'comment' not in df.columns:
        st.error("CSV에 'comment' 열이 없습니다!")
    else:
        # 3. 1단계: 주제 발견 (Topic Discovery)
        if st.button("AI가 주제 찾기 시작 🕵️"):
            with st.spinner("데이터를 분석하여 분류 기준을 수립 중..."):
                sample_comments = df['comment'].head(30).tolist()
                
                discovery_prompt = f"""
                너는 데이터 분석가야. 아래 나열된 댓글들을 읽고, 전체를 관통하는 핵심 주제를 딱 4가지로 요약해줘.
                [댓글 샘플] {sample_comments}
                [조건]
                1. 4개의 주제는 서로 겹치지 않아야 함.
                2. 출력 형식은 오직 쉼표로 구분된 단어 4개여야 함. (예: 가격, 품질, 배송, 서비스)
                3. 설명이나 다른 말은 절대 하지 마.
                """
                
                # 재시도 함수 사용
                categories = generate_with_retry(discovery_prompt)
                
                if categories == "API_ERROR":
                    st.error("주제 발견에 실패했습니다. 잠시 후 다시 시도해주세요.")
                else:
                    st.session_state.categories = categories
                    st.success(f"발견된 분류 기준: {categories}")

        # 4. 2단계: 전체 분류 (Classification)
        if "categories" in st.session_state:
            st.divider()
            st.write(f"### 2. 설정된 기준: [{st.session_state.categories}]")
            
            if st.button("전체 데이터 분류 시작 🚀"):
                # 진행률바 및 결과 컨테이너
                progress_bar = st.progress(0)
                status_text = st.empty() # 상태 메시지 표시용
                
                results = []
                total_rows = len(df)
                categories_str = st.session_state.categories

                for index, row in df.iterrows():
                    comment = row['comment']
                    
                    classify_prompt = f"""
                    다음 댓글을 아래 4가지 기준 중 하나로 분류해줘.
                    [분류 기준] {categories_str}
                    [댓글] {comment}
                    [조건] 다른 말 하지 말고 딱 분류 기준 단어 하나만 출력해.
                    """
                    
                    # 여기서 재시도 함수 호출!
                    category = generate_with_retry(classify_prompt)
                    
                    results.append(category)
                    
                    # 진행률 업데이트
                    current_progress = (index + 1) / total_rows
                    progress_bar.progress(current_progress)
                    status_text.text(f"진행 중... ({index + 1}/{total_rows})")
                
                # 결과 저장 및 표시
                df['AI_분류'] = results
                st.success("모든 분석이 완료되었습니다!")
                status_text.empty() # 상태 메시지 지우기
                
                st.write("### 최종 결과")
                st.dataframe(df)
                
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("결과 CSV 다운로드", csv, "ai_analysis_result.csv", "text/csv")
