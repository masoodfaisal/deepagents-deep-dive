"""The research brain shared by the voice notebook and its evals.

This lives outside the notebook for one reason: the eval grades the prompts below, so
the prompt that ships and the prompt that gets graded have to be the same object. If
the eval notebook re-declared `RESEARCH_INSTRUCTIONS`, the two would drift the first
time either was tuned and the eval would quietly start scoring something you no longer
run.

The model stays a caller's argument — the voice notebook and the eval may reasonably
point at different endpoints, and only the prompts need to be shared.
"""

import os

from deepagents import create_deep_agent

# A subagent that owns the searching. Giving the search tool ONLY to the subagent means
# the coordinator can't search on its own — it has to plan and then delegate. That's the
# machinery we want on screen: a todo plan, a hand-off, and the searches streaming in
# while the caller waits for an answer. The search cap keeps a live demo snappy (and the
# activity panel readable) — a broad question can otherwise trigger a dozen-plus searches.
researcher = {
    "name": "researcher",
    "description": "Searches the web on a focused question and returns concise findings.",
    "system_prompt": (
        "You are a focused web researcher. Run at most 4-5 targeted searches — no more — "
        "cross-check claims across sources, then stop and return concise findings in plain "
        "sentences. No markdown or URLs."
    ),
    # "tools": [internet_search], # TODO replace tavily tool with tools=[{"type": "web_search"}]
}

RESEARCH_INSTRUCTIONS = """You are a research coordinator answering questions that will be READ ALOUD.

ALWAYS start by calling write_todos with a short plan (2-4 steps) — every run, even an easy one, and
before any other tool call. The plan is shown live to a waiting listener, so skipping it leaves them
staring at an empty panel. Then delegate the searching to the
`researcher` subagent with the task tool, giving it complete, self-contained instructions in one call.
Update the todos as the work progresses. When the findings come back, write a short spoken report:
3-5 sentences of plain, conversational language. No markdown, no bullet lists, no URLs. Lead with the
answer; name a source only when it genuinely matters."""


def build_research_model():
    """The gateway-backed model both notebooks use for the coordinator and subagent."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("OPENAI_BASE_URL"),
        model_name="gpt-5.6-luna",
        # Deep Agents uses function tools; reasoning + tools requires /v1/responses.
        use_responses_api=True,
    )


def build_research_agent(model=None):
    """The coordinator: plans with write_todos, then delegates to `researcher`."""
    return create_deep_agent(
        model=model if model is not None else build_research_model(),
        system_prompt=RESEARCH_INSTRUCTIONS,
        subagents=[researcher],
    )
