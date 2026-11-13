"""Startup pitch generator: a multi-agent pipeline built on the OpenAI Agents SDK.

Flow: researcher (scrape + summarise, finance input guardrail) -> competitor
search -> CEO agent (skeleton -> pitch -> evaluation, one retry below the
rating threshold) -> hand-off to the sales rep (cold email + PDF).

Every agent returns a Pydantic model (structured output). Intermediate results
are collected in a per-run `PitchArtifacts` context so a UI can display them
without relying on module-level state.
"""

import os
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator

import markdown
import requests
import validators
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import (
    Agent,
    FunctionToolResult,
    GuardrailFunctionOutput,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    ToolsToFinalOutputResult,
    WebSearchTool,
    function_tool,
    input_guardrail,
    output_guardrail,
    set_tracing_disabled,
    trace,
)
from agents.exceptions import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered
from agents.model_settings import ModelSettings
from agents.tool import default_tool_error_function
from xhtml2pdf import pisa

load_dotenv(override=True)

ASSET_DIR = Path(__file__).resolve().parent / "asset"
RATING_ACCEPTABLE = 80
COMPETITOR_LIMIT = 3
# The CEO run spans skeleton, build, evaluate, one rebuild + re-evaluate, the
# handoff and the sales rep's two tool calls, which can exceed the SDK default of 10.
CEO_MAX_TURNS = 20

# One model drives every agent in a run. The first model of each vendor is the default.
LLM_VENDORS: dict[str, list[str]] = {
    "open-ai": ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-5"],
    # Checked against Gemini's OpenAI-compatible endpoint (2.x models return 404 there).
    "gemini": [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
    ],
}
GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def missing_api_key(llm_vendor: str) -> str | None:
    """Return the name of the missing env variable for a vendor, or None."""
    if llm_vendor == "gemini":
        return None if _gemini_api_key() else "GEMINI_API_KEY (or GOOGLE_API_KEY)"
    return None if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY"


def get_model(llm_vendor: str = "open-ai", model: str = LLM_VENDORS["open-ai"][0]):
    """OpenAI models are passed by name; Gemini goes through its OpenAI-compatible endpoint."""
    if llm_vendor == "gemini":
        gemini_client = AsyncOpenAI(
            base_url=os.getenv("GEMINI_BASE_URL") or GEMINI_DEFAULT_BASE_URL,
            api_key=_gemini_api_key(),
        )
        return OpenAIChatCompletionsModel(model=model, openai_client=gemini_client)
    return model


# ---------------------------------------------------------------------------
# Structured outputs
# ---------------------------------------------------------------------------

class StartupSummary(BaseModel):
    summary: str
    """Summary of the startup, summary should contain key point to be used for pitch generation in future"""


class ResearchResult(BaseModel):
    summary: str
    """Summary of the startup"""


class Competitor(BaseModel):
    name: str
    """Name of competitor"""

    detail: str
    """Detail that can be used to the startup advantage"""


class Competitors(BaseModel):
    competitors: list[Competitor]


class PitchSkeleton(BaseModel):
    reason: str
    """Your reason for why this skeleton list is the best fit and the reason why this type of pitch. how do you resolve the title"""

    skeleton: str
    """A list of sections of proposed pitch in markdown format"""

    pitch_type: str
    """Type of pitch to be generated"""

    pitch_title: str
    """Title of proposed pitch"""


class Pitch(BaseModel):
    markdown: str
    """The entire pitch in markdown"""


class Evaluation(BaseModel):
    reason: str
    """How did you come about the rating figure"""

    feedback: str
    """pitch evaluation feedback"""

    rating: float
    """The rating between 0 and 100 of how good the pitch is"""


class EmailStruct(BaseModel):
    subject: str
    """This is the subject of the cold email"""

    body: str
    """This is the HTML body of the cold email"""


class FinanceInclusionCheckOutput(BaseModel):
    finance_included: bool
    the_finance: str


@dataclass
class GuardrailCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class PitchArtifacts:
    """Everything produced during one pipeline run (passed as the run context)."""

    research: ResearchResult | None = None
    competitors: Competitors | None = None
    skeletons: list[PitchSkeleton] = field(default_factory=list)
    pitches: list[Pitch] = field(default_factory=list)
    evaluations: list[Evaluation] = field(default_factory=list)
    email: EmailStruct | None = None
    email_html: str | None = None
    pitch_pdf: bytes | None = None
    final_agent: str | None = None  # agent that ended the CEO run (Sales rep after the handoff)
    final_message: str | None = None  # plain-text final output, only set when no EmailStruct came back
    guardrail_checks: list[GuardrailCheck] = field(default_factory=list)  # every guardrail decision, in order


@dataclass
class ProgressEvent:
    stage: str  # "info" or "error" or "blocked" or "done"
    message: str
    artifacts: PitchArtifacts | None = None


def _artifacts(ctx: Any) -> PitchArtifacts | None:
    """Unwrap (possibly nested) run context wrappers down to PitchArtifacts."""
    while ctx is not None and not isinstance(ctx, PitchArtifacts):
        ctx = getattr(ctx, "context", None)
    return ctx


async def _json_output(result) -> str:
    """Agent-as-tool output extractor: return the structured output as JSON."""
    output = result.final_output
    return output.model_dump_json() if isinstance(output, BaseModel) else str(output)


def _finish_after_summary(context, results: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
    """Researcher tool_use_behavior: end the run as soon as the summariser answers.

    The prompt says each tool may be called once, but some models (e.g.
    gemini-3.8-flash) keep re-scraping until max turns. Enforcing it in code makes
    the scrape -> summarise flow deterministic for every model.
    """
    for result in results:
        if result.tool.name == "startup_summariser_tool":
            summary = StartupSummary.model_validate_json(result.output)
            return ToolsToFinalOutputResult(is_final_output=True, final_output=ResearchResult(summary=summary.summary))
    return ToolsToFinalOutputResult(is_final_output=False)


def _recording_extractor(bucket: str):
    """Agent-as-tool output extractor: store the structured output on the run
    context and hand it back to the calling agent as JSON."""

    async def extract(result) -> str:
        output = result.final_output
        artifacts = _artifacts(result.context_wrapper)
        if artifacts is not None and isinstance(output, BaseModel):
            getattr(artifacts, bucket).append(output)
        return output.model_dump_json() if isinstance(output, BaseModel) else str(output)

    return extract


# ---------------------------------------------------------------------------
# Research
# ---------------------------------------------------------------------------

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def _extract_title(soup):
    title = soup.find("title")
    return title.text.strip() if title else None


def _extract_meta_description(soup):
    meta = soup.find("meta", attrs={"name": "description"})
    return meta.get("content", "").strip() if meta else None


def _extract_headings(soup):
    headings = {}
    for i in range(1, 7):
        h_tags = soup.find_all(f"h{i}")
        if h_tags:
            headings[f"h{i}"] = [h.text.strip() for h in h_tags if h.text.strip()]
    return headings


def _extract_text_content(soup, limit=500):
    for tag in soup(["script", "style"]):
        tag.decompose()
    lines = (line.strip() for line in soup.get_text().splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    return " ".join(chunk for chunk in chunks if chunk)[:limit]


def scrape_startup_page(url: str) -> dict[str, Any]:
    try:
        response = requests.get(url, headers=BROWSER_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"url": url, "error": f"Error fetching the page: {e}"}

    soup = BeautifulSoup(response.content, "html.parser")
    return {
        "url": url,
        "status_code": response.status_code,
        "title": _extract_title(soup),
        "meta_description": _extract_meta_description(soup),
        "headings": _extract_headings(soup),
        "text_content": _extract_text_content(soup),
    }


@function_tool
def tell_me_about_startup(url: str) -> dict[str, Any]:
    """This tool gather the about us of the startup"""
    return scrape_startup_page(url)


STARTUP_SUMMARISER_INSTRUCTION = """
You will summarise what the startup is about.
The summary you generate will be verbose enough to build a pitch based off it in future.
IMPORTANT:
1. Summary must include the investor of interest.
2. Summary must not contain markdown, it must be stripped off of any textual noise.
3. Summary must contain all the details learned including the investors to attract.
"""

RESEARCHER_INSTRUCTION = """
You MUST follow these tool rules strictly:

You have access to tell_me_about_startup tool, use this tool to gather details of the startup.
You have access to summariser_tool tool, use this tool to summarise the details gathered about the startup.
You MUST NOT call any tool more than once.

NEVER use summariser_tool on anything except the output of tell_me_about_startup.

If no tool is needed at a step, you MUST respond normally without invoking a tool.

If you are unsure which tool to use, DO NOT use any tool.

Produce the final structured output.

If you violate these rules, the task fails.
"""

COMPETITOR_INSTRUCTION = f"""
You will search for the list of competitors of a startup.
Use the web search tool to perform the search, use this tool to search not more than {COMPETITOR_LIMIT} competitors of the startup.
"""

# Hosted tools such as WebSearchTool only work with OpenAI models (Responses API).
COMPETITOR_INSTRUCTION_NO_SEARCH = f"""
You will list the competitors of a startup.
Use your own knowledge of the market, list not more than {COMPETITOR_LIMIT} competitors of the startup.
"""

# ---------------------------------------------------------------------------
# Guardrail: block inputs that include financial statements
# ---------------------------------------------------------------------------

GUARDRAIL_INSTRUCTIONS = """
You are to check if startup is including financial statements
"""

# ---------------------------------------------------------------------------
# Output guardrails: rules checked in code, so no model can talk its way past them
# ---------------------------------------------------------------------------

MIN_COMPETITOR_NAME_LENGTH = 3  # skip very short names to avoid matching ordinary words


def find_competitor_mentions(text: str, competitor_names: list[str]) -> list[str]:
    """Competitor names that appear in `text` as whole words, case-insensitive."""
    mentioned = []
    for name in competitor_names:
        name = name.strip()
        if len(name) < MIN_COMPETITOR_NAME_LENGTH:
            continue
        if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text, flags=re.IGNORECASE):
            mentioned.append(name)
    return mentioned


def html_to_text(html: str) -> str:
    """Visible text of an HTML fragment with whitespace collapsed."""
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def _record_check(ctx: Any, name: str, passed: bool, detail: str) -> None:
    artifacts = _artifacts(ctx)
    if artifacts is not None:
        artifacts.guardrail_checks.append(GuardrailCheck(name=name, passed=passed, detail=detail))


@output_guardrail(name="no competitors in pitch")
async def no_competitor_guard(ctx, agent, output: Pitch) -> GuardrailFunctionOutput:
    artifacts = _artifacts(ctx)
    names = [c.name for c in artifacts.competitors.competitors] if artifacts and artifacts.competitors else []
    mentioned = find_competitor_mentions(output.markdown, names)
    detail = f"Pitch names competitor(s): {', '.join(mentioned)}" if mentioned else f"None of {len(names)} competitors named"
    _record_check(ctx, "no competitors in pitch", not mentioned, detail)
    return GuardrailFunctionOutput(output_info={"competitors_mentioned": mentioned}, tripwire_triggered=bool(mentioned))


@output_guardrail(name="addressed to Dear Investor")
async def dear_investor_guard(ctx, agent, output: EmailStruct) -> GuardrailFunctionOutput:
    addressed = "dear investor" in html_to_text(output.body).lower()
    detail = "Email is addressed to Dear Investor" if addressed else "Email body does not contain 'Dear Investor'"
    _record_check(ctx, "addressed to Dear Investor", addressed, detail)
    return GuardrailFunctionOutput(output_info={"addressed_to_investor": addressed}, tripwire_triggered=not addressed)


def pitch_builder_error(ctx, error: Exception) -> str:
    """Tool error for the CEO when the pitch builder fails.

    By default a failed agent-as-tool returns a generic "An error occurred" text.
    For a guardrail rejection, tell the CEO exactly why so it can rebuild.
    """
    if isinstance(error, OutputGuardrailTripwireTriggered):
        mentioned = error.guardrail_result.output.output_info.get("competitors_mentioned", [])
        return (
            f"Pitch rejected by guardrail: it names competitor(s) {', '.join(mentioned)}. "
            "Call pitch_builder_tool again and do not mention any competitor by name."
        )
    return default_tool_error_function(ctx, error)

# ---------------------------------------------------------------------------
# Pitch: skeleton -> build -> evaluate
# ---------------------------------------------------------------------------

PITCH_STRUCT_INSTRUCTION = """
You are an agent acting in the best interest of a startup.
You will seek a title worth the investor money.
You will seek the best type of pitch, structure and then produce the sections analysable and iterable by other agents.
You will decide the simplicity or complexity of the pitch as you seem fit.
Context:
1. You receive the summary of the startup you are to build the pitch skeleton for.
IMPORTANT:
1. skeleton must be a markdown
"""

PITCH_BUILDER_INSTRUCTION = """
You are to develop a concise pitch based on predefined sections.
The pitch you develop must sell to investors faster 'metaphorically' than gas stations.
You will receive the skeleton of the pitch you want to develop its sections.
Develop these sections religiously.
Your pitch must have a title.
IMPORTANT:
1. The pitch must not be boring.
2. The pitch result should be a markdown.
3. You MUST NOT directly include ANY COMPETITOR in the pitch.
4. If there is an evaluation that has been done on the pitch, you must use the feedback to improve the pitch.
"""

EVALUATION_INSTRUCTION = """
You are an Agent working for a startup.
You will receive the summary of the startup the pitch is generated for.
You will receive a pitch generated in markdown.
You will evaluate the correctness,viability and how captivating the pitch is.
You will further evaluate if it's standard enough for investor that suite the industry of the startup.
You can cross reference the pitch generated with the summary of the startup.
You will rate the pitch in percentage between 0 and 100.
Provide your feedback
"""

# ---------------------------------------------------------------------------
# Sales rep: cold email + PDF
# ---------------------------------------------------------------------------

DOCUMENT_CSS = """
@page { size: A4; margin: 2cm; }
body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
h2 { color: #34495e; margin-top: 20px; }
code { background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
pre { background-color: #f8f8f8; border: 1px solid #ddd; border-radius: 5px; padding: 10px; overflow-x: auto; }
blockquote { border-left: 4px solid #3498db; margin-left: 0; padding-left: 20px; color: #666; font-style: italic; }
table { border-collapse: collapse; width: 100%; margin: 20px 0; }
th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
th { background-color: #3498db; color: white; }
img { max-width: 100%; height: auto; }
"""


def _html_document(body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>{DOCUMENT_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""


def render_pitch_pdf(pitch_markdown: str) -> bytes | None:
    html_content = markdown.markdown(pitch_markdown, extensions=["extra", "codehilite", "tables", "toc"])
    buffer = BytesIO()
    status = pisa.CreatePDF(BytesIO(_html_document(html_content).encode("utf-8")), dest=buffer)
    return None if status.err else buffer.getvalue()


@function_tool
def save_pitch_as_pdf(ctx: RunContextWrapper[PitchArtifacts], pitch_markdown: str) -> dict[str, Any]:
    """save_pitch_as_pdf"""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "pitch.md").write_text(pitch_markdown, encoding="utf-8")

    pdf = render_pitch_pdf(pitch_markdown)
    if pdf is None:
        return {"success": False}
    (ASSET_DIR / "pitch.pdf").write_bytes(pdf)

    artifacts = _artifacts(ctx)
    if artifacts is not None:
        artifacts.pitch_pdf = pdf
    return {"success": True, "path": str(ASSET_DIR / "pitch.pdf")}


def write_cold_email_html(email: EmailStruct) -> str:
    """Render the cold email as an HTML document and save it to the asset folder."""
    html = _html_document(f"<h1>{email.subject}</h1>\n{email.body}")
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "pitch_cold_email.html").write_text(html, encoding="utf-8")
    return html


@function_tool
def save_cold_email_as_html(subject: str, body: str) -> dict[str, Any]:
    """save_cold_email_as_html"""
    write_cold_email_html(EmailStruct(subject=subject, body=body))
    return {"success": True, "path": str(ASSET_DIR / "pitch_cold_email.html")}


SALES_REP_INSTRUCTION = """
You are a sale representative who is in charge of writing cold emails that can trigger investors.
You must address the email as Dear Investor.
The email must be an HTML body.

CONTEXT: You will have access to a pitch markdown

IMPORTANT:
1. Use a catchy subject to the email
2. Use the tool save_cold_email_as_html to store as html
3. Use the save_pitch_as_pdf to save pitch
4. Return the final cold email as your structured output: the subject and the HTML body
"""

CEO_SYSTEM_INSTRUCTION = f"""
You are the CEO of a startup.
You will receive the summary of the startup.
You will receive the competitors of the startup.
Use pitch_skeleton_builder_tool to create skeleton of the pitch, the pitch_skeleton_builder_tool will receive a summary and return skeleton as markdown, pitch_type and pitch_title.
You have access to tool pitch_builder_tool that can build the pitch and return a markdown.
The pitch generated would have a markdown.
You have access to pitch_evaluator_tool tool to evaluate the pitch generated, the pitch_evaluator_tool will receive the summary and pitch generated.
IMPORTANT:
You should use tools only
1. After pitch markdown has been generated,
2. IF pitch evaluation falls below {RATING_ACCEPTABLE} percent rating, you must reuse the pitch_builder_tool not more than ONCE to regenerate pitch.
3. IF pitch evaluation falls below {RATING_ACCEPTABLE} percent rating, you must receive the feedback from the evaluator to regenerate and improve the pitch.
4. Handoff to sales rep agent for cold email generation and storage, pass only the pitch markdown
5. IF pitch_builder_tool returns "Pitch rejected by guardrail", call it again following that message. Rejected pitches do not count toward the rebuild limit in rule 2.
"""

# ---------------------------------------------------------------------------
# Agents: built per run so the chosen vendor and model drive every agent
# ---------------------------------------------------------------------------


@dataclass
class Pipeline:
    researcher: Agent
    competitor: Agent
    ceo: Agent


def build_pipeline(llm_vendor: str, model_name: str) -> Pipeline:
    model = get_model(llm_vendor, model_name)

    finance_guard_agent = Agent(
        name="finance guard",
        model=model,
        output_type=FinanceInclusionCheckOutput,
        instructions=GUARDRAIL_INSTRUCTIONS,
    )

    @input_guardrail
    async def finance_guard(ctx, agent, message):
        result = await Runner.run(finance_guard_agent, message, context=ctx.context)
        found = result.final_output.finance_included
        _record_check(ctx, "no financial statements in input", not found,
                      result.final_output.the_finance if found else "No financial statements found")
        return GuardrailFunctionOutput(
            output_info={"found_financial_statement_data": result.final_output},
            tripwire_triggered=result.final_output.finance_included,
        )

    summariser_agent = Agent(
        name="startup summariser",
        instructions=STARTUP_SUMMARISER_INSTRUCTION,
        output_type=StartupSummary,
        model=model,
    )

    researcher = Agent(
        name="Startup researcher",
        model=model,
        instructions=RESEARCHER_INSTRUCTION,
        tools=[
            tell_me_about_startup,
            summariser_agent.as_tool(
                tool_name="startup_summariser_tool",
                tool_description="This tool summarises the startup",
                custom_output_extractor=_json_output,
            ),
        ],
        output_type=ResearchResult,
        tool_use_behavior=_finish_after_summary,
        input_guardrails=[finance_guard],
    )

    use_web_search = llm_vendor == "open-ai"
    competitor = Agent(
        name="Scan for competitors",
        instructions=COMPETITOR_INSTRUCTION if use_web_search else COMPETITOR_INSTRUCTION_NO_SEARCH,
        model=model,
        output_type=Competitors,
        tools=[WebSearchTool(search_context_size="low")] if use_web_search else [],
    )

    pitch_skeleton_builder_agent = Agent(
        name="pitch_skeleton_builder",
        instructions=PITCH_STRUCT_INSTRUCTION,
        output_type=PitchSkeleton,
        model=model,
    )
    pitch_builder_agent = Agent(
        name="Pitch builder",
        instructions=PITCH_BUILDER_INSTRUCTION,
        output_type=Pitch,
        model=model,
        output_guardrails=[no_competitor_guard],
    )
    pitch_evaluator_agent = Agent(
        name="Pitch Evaluator",
        instructions=EVALUATION_INSTRUCTION,
        output_type=Evaluation,
        model=model,
    )

    sales_agent = Agent(
        name="Sales rep",
        instructions=SALES_REP_INSTRUCTION,
        model=model,
        tools=[save_cold_email_as_html, save_pitch_as_pdf],
        output_type=EmailStruct,
        output_guardrails=[dear_investor_guard],
        handoff_description="A sale rep that can cold email",
    )

    ceo = Agent(
        name="Startup CEO",
        model=model,
        instructions=CEO_SYSTEM_INSTRUCTION,
        tools=[
            pitch_skeleton_builder_agent.as_tool(
                tool_name="pitch_skeleton_builder_tool",
                tool_description="This tool is responsible for sketching out the elements of a pitch",
                custom_output_extractor=_recording_extractor("skeletons"),
            ),
            pitch_builder_agent.as_tool(
                tool_name="pitch_builder_tool",
                tool_description="Tool to build pitch",
                custom_output_extractor=_recording_extractor("pitches"),
                failure_error_function=pitch_builder_error,
            ),
            pitch_evaluator_agent.as_tool(
                tool_name="pitch_evaluator_tool",
                tool_description="A tool that evaluates the pitch generated",
                custom_output_extractor=_recording_extractor("evaluations"),
            ),
        ],
        handoffs=[sales_agent],
        handoff_description="An agent that can generate pitch",
        model_settings=ModelSettings(tool_choice="required"),
    )

    return Pipeline(researcher=researcher, competitor=competitor, ceo=ceo)


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
            artifacts.email_html = _html_document(f"<h1>{artifacts.email.subject}</h1>\n{artifacts.email.body}")
        yield ProgressEvent("blocked", f"Output blocked by guardrail: {name}", artifacts)
        return

    yield ProgressEvent("done", "Pitch generation completed", artifacts)
