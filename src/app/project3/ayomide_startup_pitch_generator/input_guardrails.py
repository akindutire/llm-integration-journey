"""Input guardrails: block inputs that include financial statements (LLM check)."""

from agents import Agent, GuardrailFunctionOutput, InputGuardrail, Runner, input_guardrail

from .structured_outputs import FinanceInclusionCheckOutput
from .utils import record_check

GUARDRAIL_INSTRUCTIONS = """
You are to check if startup is including financial statements
"""


def make_finance_guard(model) -> InputGuardrail:
    """Build the finance guardrail for a run; its checker agent uses the run's model."""
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
        record_check(ctx, "no financial statements in input", not found,
                     result.final_output.the_finance if found else "No financial statements found")
        return GuardrailFunctionOutput(
            output_info={"found_financial_statement_data": result.final_output},
            tripwire_triggered=found,
        )

    return finance_guard
