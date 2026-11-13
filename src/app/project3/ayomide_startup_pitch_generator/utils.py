"""Shared helpers: configuration, model selection, run-context access,
agent-as-tool output extractors and document rendering."""

import os
from io import BytesIO
from pathlib import Path
from typing import Any

import markdown
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from xhtml2pdf import pisa

from agents import FunctionToolResult, OpenAIChatCompletionsModel, ToolsToFinalOutputResult

from .structured_outputs import EmailStruct, GuardrailCheck, PitchArtifacts, ResearchResult, StartupSummary

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ASSET_DIR = Path(__file__).resolve().parent / "asset"
RATING_ACCEPTABLE = 80
COMPETITOR_LIMIT = 3

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
# Run context
# ---------------------------------------------------------------------------


def get_artifacts(ctx: Any) -> PitchArtifacts | None:
    """Unwrap (possibly nested) run context wrappers down to PitchArtifacts."""
    while ctx is not None and not isinstance(ctx, PitchArtifacts):
        ctx = getattr(ctx, "context", None)
    return ctx


def record_check(ctx: Any, name: str, passed: bool, detail: str) -> None:
    """Store a guardrail decision on the run's PitchArtifacts."""
    artifacts = get_artifacts(ctx)
    if artifacts is not None:
        artifacts.guardrail_checks.append(GuardrailCheck(name=name, passed=passed, detail=detail))


# ---------------------------------------------------------------------------
# Agent-as-tool output extractors and tool-use behaviour
# ---------------------------------------------------------------------------


def _to_text(output: Any) -> str:
    return output.model_dump_json() if isinstance(output, BaseModel) else str(output)


async def json_output(result) -> str:
    """custom_output_extractor: return the structured output as JSON."""
    return _to_text(result.final_output)


def recording_extractor(bucket: str):
    """custom_output_extractor factory: store the structured output in
    `PitchArtifacts.<bucket>` and hand it back to the calling agent as JSON."""

    async def extract(result) -> str:
        output = result.final_output
        artifacts = get_artifacts(result.context_wrapper)
        if artifacts is not None and isinstance(output, BaseModel):
            getattr(artifacts, bucket).append(output)
        return _to_text(output)

    return extract


def finish_after_summary(context, results: list[FunctionToolResult]) -> ToolsToFinalOutputResult:
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


# ---------------------------------------------------------------------------
# Document rendering
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


def html_document(body: str) -> str:
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


def html_to_text(html: str) -> str:
    """Visible text of an HTML fragment with whitespace collapsed."""
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def render_pitch_pdf(pitch_markdown: str) -> bytes | None:
    html_content = markdown.markdown(pitch_markdown, extensions=["extra", "codehilite", "tables", "toc"])
    buffer = BytesIO()
    status = pisa.CreatePDF(BytesIO(html_document(html_content).encode("utf-8")), dest=buffer)
    return None if status.err else buffer.getvalue()


def cold_email_html(email: EmailStruct) -> str:
    return html_document(f"<h1>{email.subject}</h1>\n{email.body}")


def write_cold_email_html(email: EmailStruct) -> str:
    """Render the cold email as an HTML document and save it to the asset folder."""
    html = cold_email_html(email)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "pitch_cold_email.html").write_text(html, encoding="utf-8")
    return html
