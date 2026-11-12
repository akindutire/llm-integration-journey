from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-3.8-flash"
conversation = []

class ChatResponse(BaseModel):
    role: str = Field(description="Role of the message sender, e.g. 'user' or 'assistant'")
    text: str = Field(description="The content of the reply message from the assistant")
    
def chat(message, history=None):
    # `history` lets callers (e.g. Streamlit sessions) keep their own memory;
    # defaults to the shared module-level conversation.
    conversation = history if history is not None else globals()["conversation"]
    client = genai.Client()
    conversation.append({"role": "user", "parts": [{"text": message}] })
    response = client.models.generate_content(
        model =MODEL,
        contents = conversation,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ChatResponse
        )
    )

    if isinstance(response.parsed, ChatResponse):
        parsed_res = response.parsed
    
    parsed_res = ChatResponse.model_validate_json(response.text)
    
    conversation.append({"role": "assistant", "parts": [{"text": parsed_res.text}] })
    
    return parsed_res.text