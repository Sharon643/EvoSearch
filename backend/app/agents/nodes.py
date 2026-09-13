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
    seen_urls = set()

    for query in state["sub_queries"]:
        search_results = search_web(query)

        for result in search_results:
            url = result.get("url")

            if not url or url in seen_urls:
                continue

            seen_urls.add(url)

            results.append({
                "query": query,
                "title": result.get("title"),
                "url": url,
                "content": result.get("content"),
            })

    return {
        **state,
        "search_results": results,
        "query_history": (
            state["query_history"] + state["sub_queries"]
        ),
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
SOURCE {i + 1}
Title: {result["title"]}
URL: {result["url"]}
Content: {result["content"]}
Evaluation Score: {result["score"]}
"""
        for i, result in enumerate(state["evaluated_results"])
    )

    prompt = f"""
You are a research assistant.

Answer the user's question using ONLY the provided sources.

USER QUESTION:
{state["user_query"]}

SOURCES:
{evidence}

Requirements:
- Answer the question directly.
- Use only information supported by the sources.
- Do not invent facts.
- Every major claim must include a source number like [Source 1].
- If multiple sources support a claim, cite all relevant sources.
- If the sources do not provide enough information, say so.
- Prefer higher-scoring sources when sources disagree.
- Do not include unsupported conclusions.

Structure the response clearly with:
1. A short direct answer.
2. Key findings as bullet points.
3. A brief conclusion.

Do not include a separate references section.
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

        average_relevance = sum(
            result["relevance"] for result in results
        ) / len(results)

        if (
            average_score >= 7
            and average_relevance >= 7
        ):
            decision = "answer"
        elif retry_count >= 2:
            decision = "answer"
        else:
            decision = "improve"

    return {
        **state,
        "decision": decision,
        "retry_count": (
            retry_count + 1
            if decision == "improve"
            else retry_count
        ),
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

    The previous search results were not good enough.

    Your task is to generate 3 improved search queries that find
    better information for the ORIGINAL user question.

    IMPORTANT:
    - Preserve the exact intent and subject of the original question.
    - Do NOT introduce a new industry, domain, or topic.
    - Do NOT narrow the question to healthcare, finance, education,
    manufacturing, or another domain unless the original question
    explicitly asks for it.
    - Identify what was missing or weak in the previous results.
    - Search specifically for that missing information.
    - Prefer recent and authoritative sources when the question asks
    for latest, current, or recent information.
    - Make each query meaningfully different from the previous queries.
    - Do not simply rewrite the original question.

    Return ONLY exactly 3 search queries.
    One query per line.
    Do not number them.
    Do not add explanations.
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