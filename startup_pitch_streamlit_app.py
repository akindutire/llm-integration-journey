import asyncio

import pandas as pd
import streamlit as st

from app.project3.ayomide_startup_pitch_generator.pitch_gen import orchestrate
from app.project3.ayomide_startup_pitch_generator.structured_outputs import PitchArtifacts
from app.project3.ayomide_startup_pitch_generator.utils import LLM_VENDORS, RATING_ACCEPTABLE, missing_api_key

st.set_page_config(page_title="Startup Pitch Generator", layout="wide")

# Each browser session keeps its own last result.
if "artifacts" not in st.session_state:
    st.session_state.artifacts = None

vendors = list(LLM_VENDORS)
# Default to the first vendor whose API key is configured.
default_vendor = next((v for v in vendors if not missing_api_key(v)), vendors[0])

with st.sidebar:
    st.header("Pipeline")
    # Outside the form so the model list refreshes as soon as the vendor changes.
    llm_vendor = st.selectbox("LLM vendor", vendors, index=vendors.index(default_vendor))
    model = st.selectbox("Model", LLM_VENDORS[llm_vendor])
    st.caption("The selected model runs every agent: research, competitors, pitch, evaluation and email.")
    if missing := missing_api_key(llm_vendor):
        st.warning(f"Set {missing} in .env to use {llm_vendor}.")
    if llm_vendor != "open-ai":
        st.caption("Web search is OpenAI-only; competitors come from the model's own knowledge.")
    st.write(f"Acceptable rating: {RATING_ACCEPTABLE}%")
    st.caption("Inputs containing financial statements are blocked by the finance guardrail.")
    if st.button("Clear result"):
        st.session_state.artifacts = None
        st.rerun()

st.title("Startup Pitch Generator")
st.caption("Researches a startup, scans its competitors, then writes, evaluates and packages an investor pitch.")

with st.form("startup"):
    col1, col2, col3 = st.columns(3)
    name = col1.text_input("Name")
    url = col2.text_input("Startup url", placeholder="https://example.com/about")
    industry = col3.text_input("Industry")
    more_context = st.text_area("More context that describes what you do and what you want", height=120)
    submitted = st.form_submit_button("Generate Pitch", type="primary")


async def run_pipeline(status) -> PitchArtifacts | None:
    async for event in orchestrate(name, url, industry, more_context, llm_vendor, model):
        if event.stage == "info":
            status.write(event.message)
        elif event.stage == "done":
            status.update(label=event.message, state="complete", expanded=False)
            return event.artifacts
        else:
            status.update(label=event.message, state="error", expanded=True)
            st.error(event.message)
            # Guardrail blocks still carry partial results worth showing.
            return event.artifacts
    return None


if submitted:
    if not name.strip() or not industry.strip():
        st.warning("Name and industry are required.")
    else:
        with st.status("Generating pitch...", expanded=True) as status:
            try:
                st.session_state.artifacts = asyncio.run(run_pipeline(status))
            except Exception as e:
                status.update(label="Pipeline failed", state="error")
                st.error(f"Pitch generation failed: {e}")


def render(artifacts: PitchArtifacts):
    pitch_tab, eval_tab, research_tab, skeleton_tab, email_tab = st.tabs(
        ["Pitch", "Evaluation", "Research", "Skeleton", "Cold email"]
    )

    with pitch_tab:
        if not artifacts.pitches:
            st.info("No pitch was produced.")
        else:
            final = artifacts.pitches[-1].markdown
            c1, c2 = st.columns(2)
            c1.download_button("Download markdown", final, file_name="pitch.md", mime="text/markdown")
            if artifacts.pitch_pdf:
                c2.download_button("Download PDF", artifacts.pitch_pdf, file_name="pitch.pdf", mime="application/pdf")
            st.markdown(final)
            if len(artifacts.pitches) > 1:
                with st.expander("First draft (before evaluator feedback)"):
                    st.markdown(artifacts.pitches[0].markdown)

    with eval_tab:
        st.subheader("Guardrail checks")
        if not artifacts.guardrail_checks:
            st.write("No guardrail ran.")
        for check in artifacts.guardrail_checks:
            (st.success if check.passed else st.error)(f"{'Passed' if check.passed else 'Failed'}: {check.name}. {check.detail}")
        if not artifacts.evaluations:
            st.info("No evaluation was recorded.")
        for i, evaluation in enumerate(artifacts.evaluations, start=1):
            passed = evaluation.rating >= RATING_ACCEPTABLE
            st.subheader(f"Attempt {i}")
            st.metric("Rating", f"{evaluation.rating:.0f}%", "accepted" if passed else "below threshold",
                      delta_color="normal" if passed else "inverse")
            st.markdown(f"**Reason:** {evaluation.reason}")
            st.markdown(f"**Feedback:** {evaluation.feedback}")

    with research_tab:
        st.subheader("Summary")
        st.write(artifacts.research.summary if artifacts.research else "No research summary.")
        st.subheader("Competitors")
        if artifacts.competitors and artifacts.competitors.competitors:
            st.dataframe(
                pd.DataFrame([c.model_dump() for c in artifacts.competitors.competitors]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.write("No competitors found.")

    with skeleton_tab:
        if not artifacts.skeletons:
            st.info("No skeleton was recorded.")
        for skeleton in artifacts.skeletons:
            st.subheader(skeleton.pitch_title)
            st.write(f"Pitch type: {skeleton.pitch_type}")
            st.markdown(f"**Reason:** {skeleton.reason}")
            st.markdown(skeleton.skeleton)

    with email_tab:
        if any(c.name == "addressed to Dear Investor" and not c.passed for c in artifacts.guardrail_checks[-1:]):
            st.warning("This email failed the Dear Investor guardrail. Any file the sales rep saved in asset/ may be this draft.")
        if artifacts.email is None:
            st.info(f"No cold email was produced (run ended with: {artifacts.final_agent or 'unknown'}).")
            if artifacts.final_message:
                st.write(artifacts.final_message)
        else:
            st.caption(f"Written by: {artifacts.final_agent}")
            st.write(f"Subject: {artifacts.email.subject}")
            st.download_button("Download HTML", artifacts.email_html, file_name="pitch_cold_email.html", mime="text/html")
            st.html(artifacts.email.body)


if st.session_state.artifacts is not None:
    render(st.session_state.artifacts)
