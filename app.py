import streamlit as st
import google.generativeai as genai

# 1. 페이지 설정
st.set_page_config(page_title="나만의 AI 챗봇", page_icon="🤖")
st.title("🤖 내 친구들을 위한 챗봇")
st.caption("🚀 API 키 입력 없이 바로 대화해보세요!")

# 2. API 키 설정 (입력창 삭제 -> Secrets에서 가져오기)
try:
    # Streamlit Cloud의 Secrets에 저장된 키를 불러옵니다.
    api_key = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    # 로컬이나 키 설정이 안 된 경우 에러 처리
    st.error("서버에 API 키가 설정되지 않았습니다.")
    st.stop()

# 3. 모델 설정
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# 4. 세션 상태 초기화 (대화 기록)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 이전 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. 채팅 로직
if prompt := st.chat_input("메시지를 입력하세요..."):
    # 사용자 메시지 표시
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        # AI 응답 생성
        # (간단한 구현을 위해 start_chat 대신 generate_content 사용 예시)
        # 실제로는 문맥 유지를 위해 이전 대화를 history로 변환해 넣는 것이 좋음
        
        chat = model.start_chat(history=[
            {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
            for m in st.session_state.messages[:-1] # 방금 입력한 건 제외하고 history 생성
        ])
        
        response = chat.send_message(prompt)

        # AI 메시지 표시
        with st.chat_message("assistant"):
            st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})

    except Exception as e:
        st.error(f"에러 발생: {e}")
