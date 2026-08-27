import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.state import AgentState
from app.tools.search import search_web

load_dotenv()

llm = ChatOpenAI(
    model="gpt-5.6",
    temperature=0,
)


def analyze_query(state: AgentState) -> AgentState:
    prompt = f"""
Analyze this research question:

{state["user_query"]}

Break it into 3 useful search queries.

Return ONLY the search queries, one per line.
"""

    response = llm.invoke(prompt)

    queries = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    return {
        **state,
        "sub_queries": queries[:3],
    }


def search_sources(state: AgentState) -> AgentState:
    results = []

    for query in state["sub_queries"]:
        search_results = search_web(query)

        for result in search_results:
            results.append({
                "query": query,
                "title": result.get("title"),
                "url": result.get("url"),
                "content": result.get("content"),
            })

    return {
        **state,
        "search_results": results,
    }


def generate_answer(state: AgentState) -> AgentState:
    evidence = "\n\n".join(
        f"""
SOURCE:
{result["title"]}

URL:
{result["url"]}

CONTENT:
{result["content"]}
"""
        for result in state["search_results"]
    )

    prompt = f"""
Answer the user's research question using the provided sources.

USER QUESTION:
{state["user_query"]}

SOURCES:
{evidence}

Requirements:
- Answer directly.
- Do not invent facts.
- Prefer information supported by the sources.
- Include source URLs for important claims.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
    }