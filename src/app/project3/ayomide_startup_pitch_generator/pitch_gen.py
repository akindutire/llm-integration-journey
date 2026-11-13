"""Startup pitch generator: pipeline build and orchestration.

Flow: researcher (scrape + summarise, finance input guardrail) -> competitor
search -> CEO agent (skeleton -> pitch -> evaluation, one retry below the
rating threshold) -> hand-off to the sales rep (cold email + PDF).

Module layout:
    structured_outputs.py  Pydantic outputs and the per-run PitchArtifacts context
    utils.py               config, model selection, output extractors, rendering
    tools.py               function tools (scrape, save PDF, save email)
    input_guardrails.py    finance input guardrail
    output_guardrails.py   competitor and Dear Investor output guardrails
    agents.py              instructions and agent factories
    pitch_gen.py           build_pipeline and orchestrate (this file)
"""

import os
import time
from dataclasses import dataclass
from typing import AsyncIterator

import validators

from agents import Agent, Runner, set_tracing_disabled, trace
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

from .agents import build_ceo_agent, build_competitor_agent, build_researcher_agent, build_sales_agent
from .input_guardrails import make_finance_guard
from .structured_outputs import EmailStruct, PitchArtifacts
from .utils import LLM_VENDORS, cold_email_html, get_model, missing_api_key, write_cold_email_html

# The CEO run spans skeleton, build, evaluate, one rebuild + re-evaluate, the
# handoff and the sales rep's two tool calls, which can exceed the SDK default of 10.
CEO_MAX_TURNS = 20


@dataclass
class ProgressEvent:
    stage: str  # "info" or "error" or "blocked" or "done"
    message: str
    artifacts: PitchArtifacts | None = None


# ---------------------------------------------------------------------------
# Pipeline: built per run so the chosen vendor and model drive every agent
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    researcher: Agent
    competitor: Agent
    ceo: Agent


def build_pipeline(llm_vendor: str, model_name: str) -> Pipeline:
    model = get_model(llm_vendor, model_name)
    return Pipeline(
        researcher=build_researcher_agent(model, input_guardrails=[make_finance_guard(model)]),
        competitor=build_competitor_agent(model, use_web_search=llm_vendor == "open-ai"),
        ceo=build_ceo_agent(model, sales_agent=build_sales_agent(model)),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


async def orchestrate(
    name: str,
    url: str,
    industry: str,
    more_context: str = "",
    llm_vendor: str = "open-ai",
    model: str = LLM_VENDORS["open-ai"][0],
) -> AsyncIterator[ProgressEvent]:
    """Run the full pipeline with one vendor/model for every agent, yielding
    progress events. The last event is "done" (with artifacts), "blocked"
    (guardrail tripped) or "error"."""
    if validators.url(url) is not True:
        yield ProgressEvent("error", "A valid url is required")
        return
    if llm_vendor not in LLM_VENDORS:
        yield ProgressEvent("error", f"Unknown LLM vendor: {llm_vendor}")
        return
    if missing := missing_api_key(llm_vendor):
        yield ProgressEvent("error", f"Missing API key: set {missing} in .env")
        return

    # Traces upload to the OpenAI dashboard, which needs an OpenAI key.
    set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))
    pipeline = build_pipeline(llm_vendor, model)

    startup = {"name": name, "industry": industry, "about_page_link": url, "more_context": more_context}
    artifacts = PitchArtifacts()

    try:
        with trace(f"Startup {name} Researcher and Pitch builder v{time.time()}"):
            yield ProgressEvent("info", f"Starting research with {llm_vendor} / {model}...")
            message = f"Scrap startup. Generate summary.\nContext: {startup}"
            research = await Runner.run(pipeline.researcher, message, context=artifacts)
            artifacts.research = research.final_output
            yield ProgressEvent("info", "Research completed")

            yield ProgressEvent("info", "Searching for competitors...")
            competitors = await Runner.run(pipeline.competitor, str(artifacts.research), context=artifacts)
            artifacts.competitors = competitors.final_output
            yield ProgressEvent("info", f"Found {len(artifacts.competitors.competitors)} competitors")

            yield ProgressEvent("info", "Building, evaluating and packaging the pitch...")
            message = (
                "Build pitch for startup.\n"
                f"Context: {artifacts.competitors}\n{artifacts.research}\n{more_context}"
            )
            ceo = await Runner.run(pipeline.ceo, message, context=artifacts, max_turns=CEO_MAX_TURNS)
            # After the handoff the run ends with the Sales rep, whose structured
            # output is the email. The CEO ending without a handoff yields plain text.
            artifacts.final_agent = ceo.last_agent.name
            if isinstance(ceo.final_output, EmailStruct):
                artifacts.email = ceo.final_output
                artifacts.email_html = write_cold_email_html(ceo.final_output)
            else:
                artifacts.final_message = str(ceo.final_output)
    except InputGuardrailTripwireTriggered as e:
        info = e.guardrail_result.output.output_info.get("found_financial_statement_data")
        detail = getattr(info, "the_finance", "") or "financial statements"
        yield ProgressEvent("blocked", f"Input blocked by finance guardrail: {detail}", artifacts)
        return
    except OutputGuardrailTripwireTriggered as e:
        # Only the final email reaches here; pitch rejections go back to the CEO as tool errors.
        name = e.guardrail_result.guardrail.get_name()
        artifacts.final_agent = e.guardrail_result.agent.name
        if isinstance(e.guardrail_result.agent_output, EmailStruct):
            # Keep the rejected email visible in the UI. The sales rep may already have saved its draft via the tool.
            artifacts.email = e.guardrail_result.agent_output
            artifacts.email_html = cold_email_html(artifacts.email)
        yield ProgressEvent("blocked", f"Output blocked by guardrail: {name}", artifacts)
        return

    yield ProgressEvent("done", "Pitch generation completed", artifacts)
