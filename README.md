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

# Project 3 - Startup Pitch Generator

A single agent reads a startup's about page and writes an investor pitch in markdown. This is a public demo; the full product is developed privately.

## What I Practiced

- Building an agent with the OpenAI Agents SDK: a function tool that scrapes the about page and a Pydantic structured output for the pitch
- Vendor-agnostic models: OpenAI, or Gemini through its OpenAI-compatible endpoint, chosen in the UI
- Streamlit interface with a markdown download

## Project Structure

```
src/app/project3/startup_pitch_lite/pitch_lite.py  Agent, scraping tool and model selection
startup_pitch_lite_streamlit_app.py                Streamlit interface
```

## Usage

Set `OPENAI_API_KEY` for OpenAI models or `GEMINI_API_KEY` for Gemini in `.env`.

```bash
uv run streamlit run startup_pitch_lite_streamlit_app.py
```

## Tech Stack

Python, OpenAI Agents SDK, Pydantic, BeautifulSoup, Streamlit
