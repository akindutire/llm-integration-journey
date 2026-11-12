
import asyncio
from typing import Dict, override
import requests
import json
from urllib.parse import urljoin,urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from agents import Agent,trace,Runner,function_tool,OpenAIChatCompletionsModel,input_guardrail,GuardrailFunctionOutput,WebSearchTool
from agents.model_settings import ModelSettings
from pydantic import BaseModel 
from openai import AsyncOpenAI
import os
import time
#Markdwon tools
import markdown
from xhtml2pdf import pisa
import sys
from io import BytesIO
from pathlib import Path
import validators
import gradio as gr


load_dotenv(override=True)

def getDefaultModel(llm_vendor="open-ai", modelx="gpt-4.1-mini"):
    if llm_vendor == "open-ai":
        return modelx
    elif llm_vendor == 'gemini':
       #client
        gemini_client = AsyncOpenAI(base_url= os.getenv("GEMINI_BASE_URL"), api_key=os.getenv("GEMINI_API_KEY"))
        #model
        gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client) 
        return gemini_model 
    
@function_tool
def tell_me_about_startup(url: str) -> Dict[str, str]:
    """This tool gather the about us of the startup"""
    
    # Headers to appear more like a real browser
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        # Send GET request
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract various elements
        data = {
            'url': url,
            'status_code': response.status_code,
            'title': __extract_title(soup),
            'meta_description': __extract_meta_description(soup),
            'headings': __extract_headings(soup),
            'text_content': __extract_text_content(soup),
        }
        
        return data
        
    except requests.RequestException as e:
        print(f"Error fetching the page: {e}")
        return None
    
def __extract_title(soup):
    """Extract page title"""
    title = soup.find('title')
    return title.text.strip() if title else None

def __extract_meta_description( soup):
    """Extract meta description"""
    meta = soup.find('meta', attrs={'name': 'description'})
    return meta.get('content', '').strip() if meta else None

def __extract_headings( soup):
    """Extract all headings (h1-h6)"""
    headings = {}
    for i in range(1, 7):
        h_tags = soup.find_all(f'h{i}')
        if h_tags:
            headings[f'h{i}'] = [h.text.strip() for h in h_tags if h.text.strip()]
    return headings
def __extract_text_content( soup):
    """Extract main text content"""
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = ' '.join(chunk for chunk in chunks if chunk)
    
    return text[:500]  


class StartupSummary(BaseModel):
    summary: str
    """Summary of the startup, summary should contain key point to be used for pitch generation in future"""
    
class SummariserAgent:
    def run(self):
        STARTUP_SUMMARISER_INSTRUCTION=f"""
        You will summarise what the startup is about.
        The summary you generate will be verbose enough to build a pitch based off it in future.
        IMPORTANT:
        1. Summary must include the investor of interest.
        2. Summary must not contain markdown, it must be stripped off of any textual noise.
        3. Summary must contain all the details learned including the investors to attract.
        """
        return Agent(
            name="startup summariser",
            instructions=STARTUP_SUMMARISER_INSTRUCTION,
            output_type=StartupSummary,
            model=getDefaultModel()
            )
summariser_tool = SummariserAgent().run().as_tool(tool_name="startup_summariser_tool", tool_description="This tool summarises the startup")



class ResearchResult(BaseModel):
   summary: str
   """Summary of the startup"""
   
class ResearcherAgent:
    # Crawl about page of the startup
    LIMIT = None
    
    def run(self):
        load_dotenv(override=True)
        self.LIMIT = 3
    
        RESEARCHER_INSTRUCTION=f"""
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
        return Agent(
            name="Startup researcher",
            model=getDefaultModel(),
            instructions=RESEARCHER_INSTRUCTION,
            tools=[tell_me_about_startup, summariser_tool],
            output_type=ResearchResult,
            # model_settings=ModelSettings(tool_choice="required",temperature=0),
            # input_guardrails=[finance_guard]
        )
        

class Competitor(BaseModel):
    name: str
    """Name of competitor"""
    
    detail: str
    """Detail that can be used to the startup advantage"""
   
class Competitors(BaseModel):
    competitors: list[Competitor]
 
class CompetitorListBrowserAgent:   
    
    def run(self):
        LIMIT=3
        COMPETITOR_INSTRUCTION=f"""
        You will search for the list of competitors of a startup.
        Use the web search tool to perform the search, use this tool to search not more ${LIMIT} competitors of the startup.
        """

        return Agent(
            name="Scan for competitors",
            instructions=COMPETITOR_INSTRUCTION,
            model=getDefaultModel(modelx="gpt-4o-mini"),
            output_type=Competitors,
            tools=[WebSearchTool(search_context_size='low')]
        )
        
        
        
class PitchSkeleton(BaseModel):
    reason: str
    """Your reason for why this skeleton list is the best fit and the reason why this type of pitch. how do you resolve the title""" 

    skeleton: str
    """A list of sections of proposed pitch in markdown format"""

    pitch_type: str
    """Type of pitch to be generated"""

    pitch_title: str
    """Title of proposed pitch"""
    
class PitchSkeletonBuilder:
 
    def run(self):
        PITCH_STRUCT_INSTRUCTION=f"""
        You are an agent acting in the best interest of a startup.
        You will seek a title worth the investor money.
        You will seek the best type of pitch, structure and then produce the sections analysable and iterable by other agents.
        You will decide the simplicity or complexity of the pitch as you seem fit.
        Context:
        1. You receive the summary of the startup you are to build the pitch skeleton for.
        IMPORTANT:
        1. skeleton must be a markdown
        """
        pitch_skeleton_builder = Agent(
            name = "pitch_skeleton_builder",
            instructions=PITCH_STRUCT_INSTRUCTION,
            output_type=PitchSkeleton,
            model=getDefaultModel()
        )
        return pitch_skeleton_builder
pitch_skeleton_builder_tool = PitchSkeletonBuilder().run().as_tool(tool_name="pitch_skeleton_builder_tool", tool_description="This tool is responsible for sketching out the elements of a pitch")
 

class Pitch(BaseModel):
    markdown: str
    """The entire pitch in markdown"""
class PitchBuilder:
    def run(self):
        PITCH_BUILDER_INSTRUCTION=f"""
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
        pitch_builder_agent = Agent(
            name="Pitch builder",
            instructions=PITCH_BUILDER_INSTRUCTION,
            output_type=Pitch,
            model=getDefaultModel(modelx="gpt-5")
        )
        return pitch_builder_agent
pitch_builder_tool = PitchBuilder().run().as_tool(tool_name="pitch_builder_agent", tool_description="Tool to build pitch")

class Evaluation(BaseModel):
    
    reason: str
    """How did you come about the rating figure"""
    
    feedback: str
    """pitch evaluation feedback"""  
    
    rating: float
    """The rating between 0 and 100 of how good the pitch is"""
    
class PitchEvaluator:
    
    def run(self):
        EVALAUTION_INSTRUCTION=f"""
        You are an Agent working for a startup.
        You will receive the summary of the startup the pitch is generated for.
        You will receive a pitch generated in markdown.
        You will evaluate the correctness,viability and how captivating the pitch is.
        You will further evaluate if it's standard enough for investor that suite the industry of the startup.
        You can cross reference the pitch generated with the summary of the startup.
        You will rate the pitch in percentage between 0 and 100.
        Provide your feedback
        """
        pitch_evaluator_agent = Agent(
            name="Pitch Evaluator",
            instructions=EVALAUTION_INSTRUCTION,
            model=getDefaultModel(modelx="gpt-5"),
            output_type=Evaluation
        )
        return pitch_evaluator_agent
pitch_evaluator_tool = PitchEvaluator().run().as_tool(tool_name="pitch_evaluator_tool", tool_description="A tool that evaluates the pitch egenerated")

@function_tool
def save_pitch_as_pdf(pitch_markdown: str) -> Dict[str, bool]:
    """save_pitch_as_pdf"""
    
    print(f"Pitch generated\n{pitch_markdown}")
    
    os.makedirs('./asset', exist_ok=True)
    
    with open("./asset/pitch.md", "wb") as f:
        f.write(pitch_markdown.encode('utf-8'))
  
    # Convert to HTML
    html_content = markdown.markdown(
        pitch_markdown, 
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin-left: 0;
                padding-left: 20px;
                color: #666;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Create PDF
    with open('./asset/pitch.pdf', "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(
            BytesIO(html_with_style.encode('utf-8')),
            dest=pdf_file
        )
    
    if not pisa_status.err:
        BASE_DIR = Path(__file__).resolve().parent()
        pitch_pdf = BASE_DIR / "asset" / "pitch.pdf"
        yield f"\nPitch PDF generated at {str(pitch_pdf)}"
        return {"success": True}
    else:
        print(f"Error creating PDF: {pisa_status.err}")
        return {"success": False}
  
  
@function_tool
def save_cold_email_as_html(subject: str, body: str) -> Dict[str, bool]:
    """save_cold_email_as_html"""
   
    print(f"Subject: {subject}")
   
    # Add CSS styling
    html_with_style = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 20px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            pre {{
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                overflow-x: auto;
            }}
            blockquote {{
                border-left: 4px solid #3498db;
                margin-left: 0;
                padding-left: 20px;
                color: #666;
                font-style: italic;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            img {{
                max-width: 100%;
                height: auto;
            }}
        </style>
    </head>
    <body>
        <h1>{subject}</h1>
        {body}
    </body>
    </html>
    """
    
    os.makedirs('./asset', exist_ok=True)
    #delete mail if existing
    with open('./asset/pitch_cold_email.html', 'wb') as f:
        f.write(html_with_style.encode('utf-8'))
    
    BASE_DIR = Path(__file__).resolve().parent()
    pitch_email_path = BASE_DIR / "asset" / "pitch_cold_email.html"
    yield f"Pitch generated at {str(pitch_email_path)}"
    return {'success': True}

class EmailStruct(BaseModel):
    
    subject: str
    """This is the subject of the cold email"""
    
    body: str
    """This is the HTML body of the cold email"""

class SalesAgent:
    def run(self):
        SALES_REP_INSTRUCTION=f"""
        You are a sale representative who is in charge of wriitng cold emails that can trigger investors.
        You must addresss the email as Dear Investor.
        The email must be an HTML body.\n

        CONTEXT: You will have access to a pitch markdown\n

        IMPORTANT:
        1. Use a catchy subject to the email
        2. Use the tool save_cold_email_as_html to store as html
        3. Use the save_pitch_as_pdf to save pitch
        """
        return Agent(
            name="Sales rep",
            instructions=SALES_REP_INSTRUCTION,
            model=getDefaultModel(),
            tools=[save_cold_email_as_html,
                save_pitch_as_pdf],
            handoff_description="A sale rep that can cold email"
        )
sales_agent = SalesAgent().run()

class FinanceInclusionCheckOutput(BaseModel):
    finance_included: bool
    the_finance: str
   
class FinanceInclusionCheckOutputGuardrail:
    def run(self):
        GUARDRAIL_INSTRUCTIONS=f"""
        You are to check if startup is including financial statements
        """
        return Agent(
            name="finance guard",
            model=getDefaultModel(),
            output_type=FinanceInclusionCheckOutput,
            instructions=GUARDRAIL_INSTRUCTIONS
        )
@input_guardrail
async def finance_guard(ctx, agent, message):
    result = await Runner.run(FinanceInclusionCheckOutputGuardrail().run(), message, context=ctx.context)
    finance_is_included = result.final_output.finance_included
    return GuardrailFunctionOutput(output_info={"found_financial_statement_data": result.final_output},tripwire_triggered=finance_is_included)
    
    
class CeoAgent:
    def run(self):
        
        RATING_ACCEPATABLE=80
        #Create the startup as an agent
        CEO_SYSTEM_INSTRUCTION = f"""
        You are the CEO of a startup.
        You will receive the summary of the startup.
        You will receive the competitors of the startup.
        Use pitch_skeleton_builder_tool to create skeleton of the pitch, the pitch_skeleton_builder_tool will receive a summary and return skeleton as markdown, pitch_type and pitch_title.
        You have access to tool pitch_builder_tool that can build the pitch and return a markdown.
        The pitch generated would have a markdown.
        You have access to pitch_evaluator_tool tool to evalaute the pitch generated, the pitch_evaluator_tool will receive the summary and pitch generated.
        IMPORTANT:
        You should use tools only
        1. After pitch markdown has been generared,
        2. IF pitch evaluation falls below {RATING_ACCEPATABLE} percent rating, you must reuse the pitch_builder_tool not more than ONCE to regenerate pitch.
        3. IF pitch evaluation falls below {RATING_ACCEPATABLE} percent rating, you must receive the feedback from the evaluator to regenerate and improve the pitch.
        4. Handoff to sales rep agent for cold email generation and storage, pass only the pitch markdown
        """
        
        return Agent(
            name="Startup CEO", 
            model=getDefaultModel(), 
            instructions=CEO_SYSTEM_INSTRUCTION, 
            tools=[
                pitch_skeleton_builder_tool,
                pitch_builder_tool,
                pitch_evaluator_tool,
                ],
            handoffs=[sales_agent],
            handoff_description="An agent that can generate pitch",
            model_settings=ModelSettings(tool_choice="required"),
            )
        

async def orchestrate(name, url, ind, context):
    #ensure url is a valid url
    if validators.url(url) is not True:
        yield "Error: A valid url is required"
        return
    startup = {
                "name": name,
                'industry': ind,
                'about_page_link': url,
                'more_context': context
            }
    with trace(f"Startup {startup['name']} Researcher and Pitch builder v{time.time()}"):
    
        message = f"""
            Scrap startup. Generate summary.\n
            Context: {startup}
            """
        #yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
        yield "Starting research..."
                
        research_result = await Runner.run(ResearcherAgent().run(), message)
        yield "Research completed"
        print(research_result.final_output)
        
        yield "Starting to get competitors..."
        result_competitor = await Runner.run(CompetitorListBrowserAgent().run(), str(research_result.final_output))
        yield "Competitor list fetched completed"
        print(result_competitor.final_output)
        
        more_context = startup.get('more_context')
        yield "Building pitch for startup"
        message = f"""
            Build pitch for startupy.\n
            Context: {str(result_competitor.final_output)+'\n'+str(research_result.final_output)+str(more_context)}
            """
        result2 = await Runner.run(CeoAgent().run(), message)
        yield "Startup generation completed..."
        print("-----Final-----")
        # print(result2.final_output)
if __name__ == '__main__':
   
    with gr.Blocks() as demo:
        gr.Markdown("#Your elevator pitch generator")
        with gr.Row():
            with gr.Row():
                with gr.Row(equal_height=True):
                    startup_name = gr.Textbox(lines=1, label="Name")
                    startup_url = gr.Textbox(lines=1, label="Startup url")
                    industry = gr.Textbox(lines=1, label="Industry")
                with gr.Row():
                    more_context = gr.Textbox(lines=5, label="More context that describes what you do and what you want")
                
            label = gr.Textbox(lines=20)
        button = gr.Button('Generate Pitch', variant='primary')
            
        button.click(fn=orchestrate, inputs=[startup_name, startup_url, industry, more_context], outputs=label)
        demo.launch(share=True)
    