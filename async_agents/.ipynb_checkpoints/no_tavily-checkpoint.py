from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from deepagents import create_deep_agent

load_dotenv(override=True)


tools=[{"type": "web_search"}]

agent = create_deep_agent(
    model=model,
    system_prompt=research_system_prompt,
    # tools=[search]
    tools=[{"type": "web_search"}]
)

result = agent.invoke({"messages": [{"role": "user", "content": "What are the top 2 stories in the news today?"}]})
pretty_print(result)