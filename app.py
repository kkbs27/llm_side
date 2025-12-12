import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import os
import ast  # [추가됨] 홑따옴표 리스트도 해석하는 강력한 도구

# 1. 경고 메시지 차단
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 2. 페이지 설정
st.set_page_config(page_title="AI 분석기 Final (Fix)", page_icon="⚡", layout="wide")

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

# 4. 모델 설정
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.0-flash")

# 5. 재시도 함수
def generate_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
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
st.title("⚡ AI 데이터 분석기 (Parser Fix)")
st.caption("배치 처리 + 강력한 파싱으로 '판독불가' 오류를 해결했습니다.")

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

        # 주제 탐색
        if "스스로" in analysis_mode:
            if st.button("Step 1. 주제 탐색 시작"):
                with st.spinner("주제 분석 중..."):
                    sample = df['comment'].head(20).tolist()
                    prompt = f"다음 댓글들을 읽고 핵심 주제 4가지를 쉼표로 구분해줘. 예: 맛,가격,배송,서비스 \n\n[댓글]: {sample}"
                    categories = generate_with_retry(prompt)
                    
                    if "FAIL" in categories:
                        st.error("실패했습니다.")
                    else:
                        st.session_state.final_categories = categories
                        st.success(f"✅ 발견된 주제: {categories}")
        else:
            if st.button("Step 1. 기준 설정"):
                st.session_state.final_categories = "긍정, 부정, 중립, 질문"
                st.success("✅ 기준 설정됨: 긍정, 부정, 중립, 질문")

        # --- [핵심] 고속 배치 처리 로직 ---
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
                    
                    # [수정된 프롬프트] JSON 대신 파이썬 리스트 포맷 요청 (더 안정적)
                    prompt = f"""
                    다음 {len(batch_comments)}개의 댓글을 각각 [{cats}] 중 하나로 분류해서 파이썬 리스트 형태로 줘.
                    
                    [댓글 목록]
                    {batch_comments}
                    
                    [조건]
                    1. 반드시 ['결과1', '결과2'] 형태의 파이썬 리스트만 출력해.
                    2. 설명이나 코드 블록(```) 없이 리스트만 줘.
                    3. 개수는 정확히 {len(batch_comments)}개여야 해.
                    """
                    
                    res_text = generate_with_retry(prompt)
                    
                    # [핵심 수정] 강력한 파싱 로직 (ast 사용)
                    try:
                        # 1. 앞뒤 공백 및 코드블록 제거
                        clean_text = res_text.replace("```python", "").replace("```", "").strip()
                        
                        # 2. 대괄호 [] 안에 있는 내용만 강제로 추출 (AI가 잡담 섞는 것 방지)
                        start_idx = clean_text.find('[')
                        end_idx = clean_text.rfind(']') + 1
                        
                        if start_idx != -1 and end_idx != -1:
                            clean_text = clean_text[start_idx:end_idx]
                            # 3. 파이썬 문법으로 리스트 변환 (홑따옴표, 쌍따옴표 모두 OK)
                            batch_results = ast.literal_eval(clean_text)
                        else:
                            raise ValueError("대괄호를 찾을 수 없음")

                        if len(batch_results) != len(batch_comments):
                            batch_results = ["개수오류"] * len(batch_comments)
                            
                    except Exception as e:
                        # 디버깅용: 에러 시 AI가 뭐라고 했는지 화면에 작게 출력
                        print(f"파싱 에러: {e}")
                        print(f"AI 응답: {res_text}")
                        batch_results = ["판독불가"] * len(batch_comments)
                    
                    results.extend(batch_results)
                    
                    time.sleep(1) # 1초 대기
                    
                    current_progress = min((i + BATCH_SIZE) / total_rows, 1.0)
                    progress_bar.progress(current_progress)
                    status_text.text(f"🚀 고속 분석 중... ({min(i + BATCH_SIZE, total_rows)}/{total_rows})")

                # 결과 길이 맞추기
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
