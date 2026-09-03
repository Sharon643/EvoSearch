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
You are evaluating ONE web search result.

User question:
{state["user_query"]}

Search result title:
{result["title"]}

Search result content:
{result["content"]}

Evaluate this source independently.

Score each category from 0 to 10.

RELEVANCE:
How directly does this source help answer the user's question?
- 0 = completely unrelated
- 5 = somewhat useful
- 10 = directly answers the question

AUTHORITY:
How trustworthy is the source?
- 0 = unreliable or unknown
- 5 = moderately trustworthy
- 10 = highly authoritative, such as a primary source,
  academic paper, official organization, or established publication

FRESHNESS:
How appropriate is the source's age for this specific question?
If the user asks for "latest", "current", "recent", or a year-specific
answer, recent sources should score much higher than old sources.
If the question does not depend on recency, do not heavily penalize older
but still relevant sources.

Important:
- Judge THIS source independently.
- Do not give default or identical scores.
- Do not assume a source is authoritative just because it sounds technical.
- Use the actual title and content provided.
- Do not invent information that is not present.
- Think carefully before assigning the scores.

Return ONLY three integers separated by commas.

Format:
relevance,authority,freshness

Example:
9,8,10
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

            if not all(0 <= score <= 10 for score in scores):
                continue

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
                "score": round(final_score, 2),
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

def decide_quality(state: AgentState) -> AgentState:
    results = state["evaluated_results"]
    retry_count = state["retry_count"]

    if not results:
        decision = "improve"
    else:
        average_score = sum(
            result["score"] for result in results
        ) / len(results)

        if average_score >= 7:
            decision = "answer"
        elif retry_count >= 2:
            decision = "answer"
        else:
            decision = "improve"

    return {
        **state,
        "decision": decision,
        "retry_count": retry_count + 1
        if decision == "improve"
        else retry_count,
    }