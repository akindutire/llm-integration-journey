import streamlit as st

from src.app.google_llm_chat_with_memory import MODEL, chat

st.set_page_config(page_title="Gemini Chat with Memory", layout="centered")

# Each browser session keeps its own conversation history (the model's memory).
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Session")
    st.write(f"Model: `{MODEL}`")
    st.write(f"Messages in memory: {len(st.session_state.history)}")
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.rerun()

st.title("Chat with Memory")
st.caption("Google Gemini chat that remembers earlier messages in the conversation.")

for turn in st.session_state.history:
    role = "user" if turn["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(turn["parts"][0]["text"])

if prompt := st.chat_input("Type a message"):
    with st.chat_message("user"):
        st.markdown(prompt)
    try:
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            reply = chat(prompt, history=st.session_state.history)
            st.markdown(reply)
    except Exception as e:
        # Drop the unanswered user message so history stays consistent.
        if st.session_state.history and st.session_state.history[-1]["role"] == "user":
            st.session_state.history.pop()
        st.error(f"Gemini request failed: {e}")
    else:
        st.rerun()  # refresh sidebar message count
