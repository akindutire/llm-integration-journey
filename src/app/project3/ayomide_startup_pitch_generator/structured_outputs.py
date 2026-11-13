"""Structured outputs returned by the agents, plus the per-run context that collects them."""

from dataclasses import dataclass, field

from pydantic import BaseModel


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
