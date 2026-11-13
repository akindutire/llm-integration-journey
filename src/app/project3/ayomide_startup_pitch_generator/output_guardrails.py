"""Output guardrails: rules checked in code, so no model can talk its way past them."""

import re

from agents import GuardrailFunctionOutput, output_guardrail
from agents.exceptions import OutputGuardrailTripwireTriggered
from agents.tool import default_tool_error_function

from .structured_outputs import EmailStruct, Pitch
from .utils import get_artifacts, html_to_text, record_check

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


@output_guardrail(name="no competitors in pitch")
async def no_competitor_guard(ctx, agent, output: Pitch) -> GuardrailFunctionOutput:
    artifacts = get_artifacts(ctx)
    names = [c.name for c in artifacts.competitors.competitors] if artifacts and artifacts.competitors else []
    mentioned = find_competitor_mentions(output.markdown, names)
    detail = f"Pitch names competitor(s): {', '.join(mentioned)}" if mentioned else f"None of {len(names)} competitors named"
    record_check(ctx, "no competitors in pitch", not mentioned, detail)
    return GuardrailFunctionOutput(output_info={"competitors_mentioned": mentioned}, tripwire_triggered=bool(mentioned))


@output_guardrail(name="addressed to Dear Investor")
async def dear_investor_guard(ctx, agent, output: EmailStruct) -> GuardrailFunctionOutput:
    addressed = "dear investor" in html_to_text(output.body).lower()
    detail = "Email is addressed to Dear Investor" if addressed else "Email body does not contain 'Dear Investor'"
    record_check(ctx, "addressed to Dear Investor", addressed, detail)
    return GuardrailFunctionOutput(output_info={"addressed_to_investor": addressed}, tripwire_triggered=not addressed)


def pitch_builder_error(ctx, error: Exception) -> str:
    """failure_error_function for pitch_builder_tool.

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
