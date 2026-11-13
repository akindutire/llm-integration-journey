# Project 1 - LLM Integration with Google Gemini

A practice project integrating Google Gemini into a Python application. Given a dish name and optional dietary restrictions, the app finds the best recipe for it and returns:

- Ingredients with quantities and units
- Ingredient substitutions that meet the dietary restrictions
- Step-by-step cooking instructions
- Chef's tips
- Prep and cook time, difficulty and nutrition estimates

## What I Practiced

- Calling Gemini through the `google-genai` SDK
- Prompt design for a domain-specific task
- Structured output: defining the response schema with Pydantic models and passing it to Gemini via `response_schema`, so the model returns validated, typed data instead of free text
- Building a Streamlit interface on top of the structured result

## Project Structure

```
src/app/project1/google_llm.py    Pydantic schema and Gemini call
llm_integration_streamlit_app.py  Streamlit web interface
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
echo "GOOGLE_API_KEY=your_key_here" > .env
```

## Usage

```bash
uv run streamlit run llm_integration_streamlit_app.py
```

## Tech Stack

Python, Google Gemini (`google-genai`), Pydantic, Streamlit, pandas

# Project 2 - Chat App with Memory using Google Gemini

A conversational chat app where Gemini remembers earlier messages in the conversation, so it can answer follow-up questions such as recalling the user's name.

## What I Practiced

- Multi-turn conversations: storing each user and model message and sending the full history to Gemini on every request
- Structured output: each reply is parsed into a Pydantic `ChatResponse` model via `response_schema`
- Per-session memory in Streamlit using `st.session_state`, with an option to clear the conversation

## Project Structure

```
src/app/project2/google_llm_chat_with_memory.py  Chat logic, memory and response schema
chat_with_memory_streamlit_app.py       Streamlit chat interface
main.py                                 Command-line memory demo
```

## Usage

Uses the same setup as Project 1.

Web app:

```bash
uv run streamlit run chat_with_memory_streamlit_app.py
```

Command line:

```bash
uv run python main.py
```

## Tech Stack

Python, Google Gemini (`google-genai`), Pydantic, Streamlit

# Project 3 - Startup Pitch Generator (Multi-Agent)

Given a startup's name, about page url, industry and extra context, a team of agents researches the startup, finds its competitors, then writes, evaluates and packages an investor pitch with a cold email.

## What I Practiced

- Multi-agent orchestration with the OpenAI Agents SDK: agents as tools, hand-offs and tracing
- Vendor-agnostic agents: the vendor (OpenAI or Gemini) and one model for every agent are chosen in the UI; Gemini runs through its OpenAI-compatible endpoint
- Structured output: every agent returns a Pydantic model (summary, competitors, skeleton, pitch, evaluation)
- Evaluation loop: an evaluator agent rates the pitch; below 80% the CEO agent rebuilds it once using the feedback
- Guardrails: a finance input guardrail (LLM check), plus output guardrails checked in code: the pitch must not name a competitor (rejected pitches go back to the CEO to rebuild) and the email must be addressed to Dear Investor
- Streamlit interface showing progress, the pitch, evaluation scores, research and the cold email, with markdown, PDF and HTML downloads

## Project Structure

```
src/app/project3/ayomide_startup_pitch_generator/pitch_generator.py  Agents, schemas, guardrail and pipeline
src/app/project3/ayomide_startup_pitch_generator/pitch_gen.ipynb     Exploration notebook
startup_pitch_streamlit_app.py                                       Streamlit interface
```

## Usage

Set `OPENAI_API_KEY` for OpenAI models or `GEMINI_API_KEY` / `GOOGLE_API_KEY` for Gemini in `.env`. Web search for competitors is only available with OpenAI; with Gemini the model lists competitors from its own knowledge.

```bash
uv run streamlit run startup_pitch_streamlit_app.py
```

## Tech Stack

Python, OpenAI Agents SDK, Pydantic, BeautifulSoup, xhtml2pdf, Streamlit, pandas
