import asyncio

import streamlit as st
import validators

from app.project3.startup_pitch_lite.pitch_lite import DEFAULT_MODELS, generate_pitch, missing_api_key

st.set_page_config(page_title="Startup Pitch Generator", layout="centered")

if "pitch" not in st.session_state:
    st.session_state.pitch = None

with st.sidebar:
    llm_vendor = st.selectbox("LLM vendor", list(DEFAULT_MODELS))
    model = st.text_input("Model", value=DEFAULT_MODELS[llm_vendor])
    if missing := missing_api_key(llm_vendor):
        st.warning(f"Set {missing} in .env to use {llm_vendor}.")

st.title("Startup Pitch Generator")
st.caption("Demo: one agent reads a startup's about page and writes an investor pitch.")

with st.form("startup"):
    name = st.text_input("Name")
    url = st.text_input("About page url", placeholder="https://example.com/about")
    industry = st.text_input("Industry")
    submitted = st.form_submit_button("Generate pitch", type="primary")

if submitted:
    if not name.strip() or not industry.strip() or validators.url(url) is not True:
        st.warning("Name, industry and a valid url are required.")
    else:
        with st.spinner("Writing pitch..."):
            try:
                st.session_state.pitch = asyncio.run(generate_pitch(name, url, industry, llm_vendor, model))
            except Exception as e:
                st.error(f"Pitch generation failed: {e}")

if st.session_state.pitch is not None:
    pitch = st.session_state.pitch
    st.download_button("Download markdown", pitch.markdown, file_name="pitch.md", mime="text/markdown")
    st.markdown(pitch.markdown)
