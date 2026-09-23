from dotenv import load_dotenv
from langchain_core.tools import tool
from deepagents import create_deep_agent

from util import pretty_print, print_todos, print_exchange, print_activity
from langchain_core.tools import tool
from tavily import TavilyClient
from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
import os

load_dotenv(override=True)

model = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL"),
    model_name="gpt-5.5",
    # model_name="gpt-5.6-luna",
    # Deep Agents uses function tools; reasoning + tools requires /v1/responses.
    # use_responses_api=True,
    streaming=False,

)


research_system_prompt = """You are a travel research assistant.
Use the search tool to investigate the destination or travel question.
Return a concise briefing with clear headings and inline citations.
Limit yourself to 3 search calls."""

tools=[{"type": "web_search"}]

graph = create_deep_agent(
    model=model,
    system_prompt=research_system_prompt,
    # tools=[search]
    tools=tools
)

#result = agent.invoke({"messages": [{"role": "user", "content": "What are the top 2 stories in the news today?"}]})
#pretty_print(result)
