import streamlit as st
import google.generativeai as genai
import pandas as pd
import time
import os
import json

# 1. 보기 싫은 경고 메시지 차단
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

# 2. 페이지 설정
st.set_page_config(page_title="AI 분석기 Final (Fast)", page_icon="⚡", layout="wide")

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

# 4. 모델 설정 (Gemini 2.0 Flash 사용)
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
            print(f"Error: {error_msg}")
            
            if "429" in error_msg or "Quota" in error_msg:
                with st.sidebar:
                    st.toast(f"🚦 잠시 대기 중... ({attempt+1}/{max_retries})")
                time.sleep(5) # 에러나면 5초 대기
            else:
                time.sleep(1)
    return "FAIL"

# --- 메인 UI 시작 ---
st.title("⚡ AI 데이터 분석기 (Speed Up Ver.)")
st.caption("배치 처리(Batch Processing) 기술을 적용하여 속도를 10배 높였습니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file is not None:
    # 인코딩 자동 감지 로직
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

        # 주제 탐색 로직 (기존과 동일)
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
                
                # 배치 사이즈 설정 (한 번에 10개씩 처리)
                BATCH_SIZE = 10 
                target_df = df
                total_rows = len(target_df)
                cats = st.session_state.final_categories

                # 배치 단위로 반복
                for i in range(0, total_rows, BATCH_SIZE):
                    # 10개씩 자르기
                    batch = target_df.iloc[i : i + BATCH_SIZE]
                    batch_comments = batch['comment'].tolist()
                    
                    # 프롬프트 구성 (JSON 형태로 요청하여 파싱 정확도 높임)
                    prompt = f"""
                    다음 {len(batch_comments)}개의 댓글을 각각 [{cats}] 중 하나로 분류해줘.
                    
                    [댓글 목록]
                    {json.dumps(batch_comments, ensure_ascii=False)}
                    
                    [조건]
                    1. 결과는 반드시 ["결과1", "결과2", ...] 형태의 JSON 리스트로만 출력해.
                    2. 다른 말은 절대 하지 마. 오직 리스트만 출력해.
                    3. 개수는 정확히 {len(batch_comments)}개여야 해.
                    """
                    
                    # API 호출
                    res_text = generate_with_retry(prompt)
                    
                    # 결과 파싱 (JSON -> 리스트 변환)
                    try:
                        # 코드 블록 기호가 있으면 제거
                        res_text = res_text.replace("```json", "").replace("```", "").strip()
                        batch_results = json.loads(res_text)
                        
                        # 개수가 맞는지 확인
                        if len(batch_results) != len(batch_comments):
                            # 개수 안 맞으면 에러 처리 대신 '에러'라고 채움
                            batch_results = ["에러"] * len(batch_comments)
                            
                    except:
                        # 파싱 실패 시
                        batch_results = ["판독불가"] * len(batch_comments)
                    
                    results.extend(batch_results)
                    
                    # [중요] 10개 처리하고 1초만 쉼 (기존: 1개 처리하고 2초 쉼 -> 속도 약 20배 향상)
                    time.sleep(1)
                    
                    # 진행률 업데이트
                    current_progress = min((i + BATCH_SIZE) / total_rows, 1.0)
                    progress_bar.progress(current_progress)
                    status_text.text(f"🚀 고속 분석 중... ({min(i + BATCH_SIZE, total_rows)}/{total_rows})")

                # 결과 길이 맞추기 (혹시 모를 에러 방지)
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
