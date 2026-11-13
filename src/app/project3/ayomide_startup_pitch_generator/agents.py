"""Agent definitions: instructions and one factory per agent.

Agents are built per run (not at import) so the vendor/model chosen in the UI
drives every agent. `from agents import ...` below is the OpenAI Agents SDK:
this module is imported as part of the package, so its own name does not
shadow the SDK.
"""

from agents import Agent, InputGuardrail, WebSearchTool
from agents.model_settings import ModelSettings

from .output_guardrails import dear_investor_guard, no_competitor_guard, pitch_builder_error
from .structured_outputs import (
    Competitors,
    EmailStruct,
    Evaluation,
    Pitch,
    PitchSkeleton,
    ResearchResult,
    StartupSummary,
)
from .tools import save_cold_email_as_html, save_pitch_as_pdf, tell_me_about_startup
from .utils import COMPETITOR_LIMIT, RATING_ACCEPTABLE, finish_after_summary, json_output, recording_extractor

# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

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
# Research agents
# ---------------------------------------------------------------------------


def build_summariser_agent(model) -> Agent:
    return Agent(
        name="startup summariser",
        instructions=STARTUP_SUMMARISER_INSTRUCTION,
        output_type=StartupSummary,
        model=model,
    )


def build_researcher_agent(model, input_guardrails: list[InputGuardrail]) -> Agent:
    return Agent(
        name="Startup researcher",
        model=model,
        instructions=RESEARCHER_INSTRUCTION,
        tools=[
            tell_me_about_startup,
            build_summariser_agent(model).as_tool(
                tool_name="startup_summariser_tool",
                tool_description="This tool summarises the startup",
                custom_output_extractor=json_output,
            ),
        ],
        output_type=ResearchResult,
        tool_use_behavior=finish_after_summary,
        input_guardrails=input_guardrails,
    )


def build_competitor_agent(model, use_web_search: bool) -> Agent:
    return Agent(
        name="Scan for competitors",
        instructions=COMPETITOR_INSTRUCTION if use_web_search else COMPETITOR_INSTRUCTION_NO_SEARCH,
        model=model,
        output_type=Competitors,
        tools=[WebSearchTool(search_context_size="low")] if use_web_search else [],
    )


# ---------------------------------------------------------------------------
# Pitch agents (used by the CEO as tools)
# ---------------------------------------------------------------------------


def build_pitch_skeleton_builder_agent(model) -> Agent:
    return Agent(
        name="pitch_skeleton_builder",
        instructions=PITCH_STRUCT_INSTRUCTION,
        output_type=PitchSkeleton,
        model=model,
    )


def build_pitch_builder_agent(model) -> Agent:
    return Agent(
        name="Pitch builder",
        instructions=PITCH_BUILDER_INSTRUCTION,
        output_type=Pitch,
        model=model,
        output_guardrails=[no_competitor_guard],
    )


def build_pitch_evaluator_agent(model) -> Agent:
    return Agent(
        name="Pitch Evaluator",
        instructions=EVALUATION_INSTRUCTION,
        output_type=Evaluation,
        model=model,
    )


# ---------------------------------------------------------------------------
# Sales rep and CEO
# ---------------------------------------------------------------------------


def build_sales_agent(model) -> Agent:
    return Agent(
        name="Sales rep",
        instructions=SALES_REP_INSTRUCTION,
        model=model,
        tools=[save_cold_email_as_html, save_pitch_as_pdf],
        output_type=EmailStruct,
        output_guardrails=[dear_investor_guard],
        handoff_description="A sale rep that can cold email",
    )


def build_ceo_agent(model, sales_agent: Agent) -> Agent:
    return Agent(
        name="Startup CEO",
        model=model,
        instructions=CEO_SYSTEM_INSTRUCTION,
        tools=[
            build_pitch_skeleton_builder_agent(model).as_tool(
                tool_name="pitch_skeleton_builder_tool",
                tool_description="This tool is responsible for sketching out the elements of a pitch",
                custom_output_extractor=recording_extractor("skeletons"),
            ),
            build_pitch_builder_agent(model).as_tool(
                tool_name="pitch_builder_tool",
                tool_description="Tool to build pitch",
                custom_output_extractor=recording_extractor("pitches"),
                failure_error_function=pitch_builder_error,
            ),
            build_pitch_evaluator_agent(model).as_tool(
                tool_name="pitch_evaluator_tool",
                tool_description="A tool that evaluates the pitch generated",
                custom_output_extractor=recording_extractor("evaluations"),
            ),
        ],
        handoffs=[sales_agent],
        handoff_description="An agent that can generate pitch",
        model_settings=ModelSettings(tool_choice="required"),
    )
