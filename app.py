import streamlit as st
import google.generativeai as genai

# 1. 페이지 설정 (제목 및 아이콘)
st.set_page_config(page_title="나만의 AI 챗봇", page_icon="🤖")
st.title("🤖 Gemini Pro 챗봇")
st.caption("🚀 Streamlit으로 만든 나만의 LLM 앱")

# 2. 사이드바에서 API 키 입력받기 (보안)
with st.sidebar:
    api_key = st.text_input("Google API Key", type="password")
    st.markdown("API 키를 입력하면 대화가 시작됩니다.")

# 3. 세션 상태 초기화 (대화 기록 저장용)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. 화면에 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요..."):
    # API 키가 없으면 경고
    if not api_key:
        st.error("왼쪽 사이드바에 API 키를 먼저 입력해주세요!")
        st.stop()

    # 사용자 메시지 화면 표시 및 저장
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # 모델 설정
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-pro-latest")

        # Gemini 포맷에 맞춰 대화 기록 변환 (Context 유지)
        gemini_history = [
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in st.session_state.messages
        ]
        
        # 실제로는 마지막 메시지를 제외하고 history에 넣고, 마지막 메시지는 send_message로 보냄
        # 간단한 구현을 위해 여기서는 generate_content로 전체 맥락을 리스트로 주는 방식 활용 가능하지만,
        # 정확도를 위해 start_chat 방식을 추천함. 아래는 간소화된 로직임.
        
        chat = model.start_chat(history=[]) 
        # (주의: 실제 앱에서는 토큰 제한 등을 고려해 history 길이를 조절해야 함)
        
        response = chat.send_message(prompt) # 여기선 단순화를 위해 현재 질문만 보냄 (이전 맥락 필요시 위 history 변수 활용)

        # AI 응답 화면 표시 및 저장
        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"에러 발생: {e}")