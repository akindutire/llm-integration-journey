"""Startup pitch generator (public demo).

One agent reads a startup's about page and writes an investor pitch in markdown.
This is a minimal demo of the OpenAI Agents SDK with a function tool and a
structured output.
"""

import os
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel

from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, set_tracing_disabled

load_dotenv(override=True)

DEFAULT_MODELS = {"open-ai": "gpt-4.1-mini", "gemini": "gemini-3.5-flash"}
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

INSTRUCTIONS = """
You write investor pitches for startups.
Use tell_me_about_startup to read the startup's about page, then write a pitch in markdown with a title.
"""


class Pitch(BaseModel):
    title: str
    markdown: str


def _gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def missing_api_key(llm_vendor: str) -> str | None:
    """Name of the env variable the vendor needs, or None when it is set."""
    if llm_vendor == "gemini":
        return None if _gemini_api_key() else "GEMINI_API_KEY"
    return None if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY"


def get_model(llm_vendor: str, model: str):
    if llm_vendor == "gemini":
        client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=_gemini_api_key())
        return OpenAIChatCompletionsModel(model=model, openai_client=client)
    return model


@function_tool
def tell_me_about_startup(url: str) -> dict[str, Any]:
    """Fetch the startup's about page and return its title and visible text."""
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return {"url": url, "error": str(e)}

    soup = BeautifulSoup(response.content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    title = soup.find("title")
    return {
        "url": url,
        "title": title.text.strip() if title else None,
        "text": " ".join(soup.get_text().split())[:1500],
    }


async def generate_pitch(name: str, url: str, industry: str, llm_vendor: str = "open-ai", model: str = "") -> Pitch:
    set_tracing_disabled(not os.getenv("OPENAI_API_KEY"))
    agent = Agent(
        name="Pitch writer",
        instructions=INSTRUCTIONS,
        model=get_model(llm_vendor, model or DEFAULT_MODELS[llm_vendor]),
        tools=[tell_me_about_startup],
        output_type=Pitch,
    )
    message = f"Startup: {name}\nIndustry: {industry}\nAbout page: {url}"
    result = await Runner.run(agent, message)
    return result.final_output
