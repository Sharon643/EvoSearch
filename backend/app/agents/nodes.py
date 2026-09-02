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
    scored_results = []

    for result in state["search_results"]:
        prompt = f"""
Evaluate this web search result for the user's question.

User question:
{state["user_query"]}

Title:
{result["title"]}

Content:
{result["content"]}

Give scores from 0 to 10 for:

Relevance: How directly does this result help answer the question?
Authority: How trustworthy is the source?
Freshness: How recent/useful is the information?

Return ONLY this format:
relevance,authority,freshness

Example:
8,9,7
"""

        response = llm.invoke(prompt)

        try:
            scores = [
                int(score.strip())
                for score in response.content.split(",")
            ]

            if len(scores) != 3:
                continue

            relevance, authority, freshness = scores

            final_score = (
                relevance * 0.5
                + authority * 0.3
                + freshness * 0.2
            )

            scored_results.append({
                **result,
                "relevance": relevance,
                "authority": authority,
                "freshness": freshness,
                "score": final_score,
            })

        except (ValueError, TypeError):
            continue

    scored_results.sort(
        key=lambda result: result["score"],
        reverse=True,
    )

    return {
        **state,
        "evaluated_results": scored_results[:5],
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