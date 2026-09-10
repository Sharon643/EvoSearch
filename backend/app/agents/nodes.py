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
    results = state["search_results"][:10]

    formatted_results = "\n\n".join(
        f"""
RESULT {i + 1}
Title: {result["title"]}
Content: {result["content"]}
"""
        for i, result in enumerate(results)
    )

    prompt = f"""
You are evaluating web search results.

User question:
{state["user_query"]}

Evaluate ALL of the following results independently.

{formatted_results}

For each result, give scores from 0 to 10:

Relevance:
How directly does the result help answer the user's question?

Authority:
How trustworthy is the source?

Freshness:
How appropriate is the source's age for this question?
For questions asking for latest, current, or recent information,
recent sources should score higher.

Return ONLY one line per result using this format:

RESULT_NUMBER,relevance,authority,freshness

Example:

1,9,8,10
2,6,7,8
3,8,9,9

Do not add explanations.
"""

    response = llm.invoke(prompt)

    scored_results = []

    try:
        for line in response.content.splitlines():
            parts = line.strip().split(",")

            if len(parts) != 4:
                continue

            result_number, relevance, authority, freshness = [
                int(value.strip())
                for value in parts
            ]

            if not 1 <= result_number <= len(results):
                continue

            if not all(0 <= score <= 10 for score in [
                relevance,
                authority,
                freshness,
            ]):
                continue

            result = results[result_number - 1]

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
        pass

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

def improve_query(state: AgentState) -> AgentState:
    results = state["evaluated_results"]

    sources = "\n\n".join(
        f"""
Title: {result["title"]}
Content: {result["content"]}
Score: {result["score"]}
"""
        for result in results
    )

    prompt = f"""
You are improving a web search strategy.

Original user question:
{state["user_query"]}

Previous search queries:
{state["sub_queries"]}

Previous evaluated results:
{sources}

The previous search was not good enough.

Identify what information is missing or weak in the previous results,
then generate exactly 3 better search queries.

Rules:
- Return ONLY the 3 queries.
- One query per line.
- Do not number them.
- Do not add explanations.
- Make the queries different from the previous queries.
- Focus on information missing from the previous results.
- Prefer specific queries over broad queries.
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