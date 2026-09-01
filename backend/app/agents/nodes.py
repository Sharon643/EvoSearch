import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.agents.state import AgentState
from app.tools.search import search_web

load_dotenv()

llm = ChatOpenAI(
    model="llama3:latest",
    base_url="http://localhost:11434/v1",
    api_key="ollama",
    temperature=0,
)


def analyze_query(state: AgentState) -> AgentState:
    prompt = f"""
You are a search query planner.

User question:
{state["user_query"]}

Generate exactly 3 search queries that would help answer the question.

Rules:
- Return ONLY the 3 queries.
- One query per line.
- Do not number them.
- Do not add explanations.
- Do not use quotes.
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

def evaluate_sources(state: AgentState) -> AgentState:
    results_text = "\n\n".join(
        f"Title: {r['title']}\n"
        f"URL: {r['url']}\n"
        f"Content: {r['content']}"
        for r in state["search_results"]
    )

    prompt = f"""
You are evaluating web search results.

User question:
{state["user_query"]}

Search results:
{results_text}

Select the most useful results for answering the question.

Return ONLY the numbers of useful results,
one per line.

Choose at most 5.
"""

    response = llm.invoke(prompt)

    selected = []

    for line in response.content.splitlines():
        line = line.strip()

        if line.isdigit():
            index = int(line) - 1

            if 0 <= index < len(state["search_results"]):
                selected.append(state["search_results"][index])

    return {
        **state,
        "evaluated_results": selected,
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
        for result in state["evaluated_results"]
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