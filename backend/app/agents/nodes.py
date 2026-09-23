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
You are the research planning component of an AI research agent.

USER QUESTION:
{state["user_query"]}

Your task is to:

1. Identify 3 to 5 important research dimensions.
2. Create exactly 3 search queries.
3. Distribute the research dimensions across those 3 queries.

For broad questions about latest, current, recent, or emerging trends,
prioritize breadth of coverage.

For AI engineering questions, relevant dimensions may include:

- AI applications and agents
- Model and LLM engineering
- Evaluation and reliability
- Data and context engineering
- Infrastructure and deployment
- AI-assisted software engineering
- Observability and operations

Only include dimensions relevant to the user's question.

IMPORTANT QUERY DISTRIBUTION RULE:

Every research-plan dimension must be assigned to at least one query.

Distribute the dimensions across the three queries.

For example, if there are five dimensions:

Query 1 → AI applications and agents + Model and LLM engineering
Query 2 → Evaluation and reliability + Data and context engineering
Query 3 → Infrastructure and deployment

Do NOT leave a research-plan dimension without a query.

Do NOT make all three queries focus on the same topic.

Each query should investigate a different group of dimensions.

QUERY RULES:

- Generate exactly 3 queries.
- Each query must contain 5 to 12 words.
- Preserve the user's original intent.
- Make queries specific enough to retrieve useful sources.
- Use the assigned research dimensions when constructing each query.
- For latest/current/recent questions, prioritize recent information.
- Do not add unnecessary phrases such as:
  "with a focus on"
  "according to leading experts"
  "research papers and industry reports"
- Do not invent specific years unless useful.
- Do not number the queries.
- Do not add explanations.

Return ONLY:

PLAN:
aspect 1
aspect 2
aspect 3
aspect 4
aspect 5

QUERIES:
query 1
query 2
query 3
"""

    response = llm.invoke(prompt)

    lines = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    research_plan = []
    queries = []

    mode = None

    for line in lines:

        if line.upper() == "PLAN:":
            mode = "plan"
            continue

        if line.upper() == "QUERIES:":
            mode = "queries"
            continue

        if mode == "plan":
            # Handle comma-separated plan items
            parts = [
                part.strip()
                for part in line.split(",")
                if part.strip()
            ]

            research_plan.extend(parts)

        elif mode == "queries":
            queries.append(line)

    return {
        **state,
        "research_plan": research_plan[:5],
        "sub_queries": queries[:3],
    }
def search_sources(state: AgentState) -> AgentState:
    results = []
    seen_urls = set()

    for query in state["sub_queries"]:
        search_results = search_web(query)

        query_count = 0

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

            query_count += 1

            if query_count >= 5:
                break

    return {
        **state,
        "search_results": results,
        "evaluated_results": [],
        "evidence_summary": "",
        "final_answer": "",
        "validation": "",
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
    You are evaluating web search results for a research agent.

    USER QUESTION:
    {state["user_query"]}

    Evaluate ALL results independently.

    {formatted_results}

    For each result, give scores from 0 to 10.

    RELEVANCE:
    How directly does this source answer the user's exact question?
    - 9-10 = directly answers the question
    - 7-8 = strongly useful
    - 4-6 = somewhat related but indirect
    - 0-3 = mostly unrelated

    AUTHORITY:
    How trustworthy is the source?
    - 9-10 = highly authoritative primary source, major research institution,
    established technology company, or peer-reviewed/reputable research
    - 7-8 = generally credible
    - 4-6 = questionable or secondary source
    - 0-3 = unreliable

    FRESHNESS:
    How appropriate is the source's age for this question?
    For questions asking for latest, current, or recent information,
    recent sources should score higher.

    IMPORTANT:
    - Judge relevance against the USER QUESTION, not merely whether the
    source discusses AI.
    - Generic AI articles should receive a lower relevance score.
    - A source should not receive a high relevance score just because it
    contains keywords from the question.
    - Prefer sources that provide concrete evidence, trends, measurements,
    technical developments, or specific findings.
    - Do not infer information that is not present in the source.

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

def synthesize_evidence(state: AgentState) -> AgentState:
    results = state["evaluated_results"]

    sources = "\n\n".join(
        f"""
SOURCE {i + 1}
Query: {result["query"]}
Title: {result["title"]}
Content: {result["content"]}
Score: {result["score"]}
Relevance: {result["relevance"]}
Authority: {result["authority"]}
Freshness: {result["freshness"]}
"""
        for i, result in enumerate(results)
    )

    prompt = f"""
You are the evidence synthesis component of a research agent.

USER QUESTION:
{state["user_query"]}

RESEARCH PLAN:
{state["research_plan"]}

EVALUATED SOURCES:
{sources}

Your job is to identify the strongest evidence that directly answers
the user's question.

IMPORTANT:

The research plan is ONLY a guide for organizing research.

Do NOT try to fill every research-plan aspect.

Only include findings that are genuinely supported by the sources.

For a question asking about "latest AI engineering trends", a valid
finding should represent a meaningful engineering trend, development,
practice, technology, or shift in how AI systems are built, evaluated,
deployed, or operated.

DO NOT treat the following as trends by themselves:

- an isolated product
- a company example
- a statistic
- a use case
- a general description of AI capabilities
- a prediction without supporting evidence
- an old/general explanation of AI
- a technology merely being mentioned
- a claim that something is "important" without evidence of a trend

RULES:

1. Use ONLY information present in the sources.

2. Do not use general knowledge.

3. Do not invent facts.

4. Do not exaggerate what a source says.

5. Do not combine unrelated claims into a trend.

6. Combine multiple sources when they support the same trend.

7. Prefer trends supported by multiple independent sources.

8. A single source may support a finding if the evidence is direct
   and the source is authoritative.

9. Each finding may optionally be assigned to ONE research-plan aspect.

10. Only assign an aspect when the connection is explicit and strong.

11. Never force a finding into an aspect.

12. Never create a new research-plan aspect.

13. Never duplicate the same finding.

14. If the evidence is insufficient, say so instead of inventing
    additional findings.

For every genuine finding, use:

THEME:
Short name of the trend.

RESEARCH PLAN ASPECT:
One exact aspect from the research plan, or:
NONE

FINDING:
What the sources directly support.

SOURCES:
Source numbers.

CONFIDENCE:
HIGH, MEDIUM, or LOW

After the findings, provide:

EVIDENCE COVERAGE:

For each research-plan aspect, state:

<aspect>: SUPPORTED

or

<aspect>: INSUFFICIENT EVIDENCE

IMPORTANT:

An aspect is SUPPORTED ONLY when a finding above explicitly assigns
it to that exact aspect.

If no finding explicitly assigns an aspect, it MUST be:
INSUFFICIENT EVIDENCE.

Do not mark an aspect as supported merely because a source is
generally related to it.

Return ONLY the synthesized evidence.
Do not explain your reasoning.
"""

    response = llm.invoke(prompt)

    return {
        **state,
        "evidence_summary": response.content,
    }

def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""
    You are a research assistant.

    Answer the user's question using ONLY the synthesized evidence below.

    USER QUESTION:
    {state["user_query"]}

    RESEARCH PLAN:
    {state["research_plan"]}

    SYNTHESIZED EVIDENCE:
    {state["evidence_summary"]}

    Requirements:

    1. Answer the user's question directly.

    2. Use only claims supported by the synthesized evidence.

    3. Every major factual claim must include source numbers such as
    [Source 1] or [Sources: 1, 2].

    4. Do not invent facts.

    5. Do not exaggerate the evidence.

    6. Do not claim that the identified findings represent ALL current
    trends unless the evidence explicitly establishes that.

    7. Prefer wording such as:
    "The research identified..."
    "The available evidence points to..."
    "Among the sources analyzed..."

    8. Do NOT use phrases such as:
    "The latest trends are..."
    "The most important trend is..."
    "The industry is moving toward..."
    unless the evidence explicitly supports those claims.

    9. Do not rank the trends.

    10. Do not turn closely related findings into separate trends.

    11. If the available evidence is limited, explicitly say that the
        findings represent only the areas supported by the retrieved
        sources.

    12. If an important research-plan aspect lacks evidence, mention that
        briefly.

    Structure:

    1. Short direct answer.
    2. Key findings as bullet points.
    3. Brief evidence limitation, if necessary.
    4. Short conclusion.

    Do not include a separate references section.
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "final_answer": response.content,
    }
def validate_answer(state: AgentState) -> AgentState:
    sources = "\n\n".join(
        f"""
SOURCE {i + 1}
Title: {result["title"]}
Content: {result["content"]}
"""
        for i, result in enumerate(state["evaluated_results"])
    )

    prompt = f"""
You are validating a research answer.

USER QUESTION:
{state["user_query"]}

ANSWER:
{state["final_answer"]}

AVAILABLE SOURCES:
{sources}

Check whether the answer is properly supported by the sources.

Check:
1. Does every [Source N] reference an existing source?
2. Does each cited source actually support the claim it is attached to?
3. Are there important factual claims without citations?
4. Does the answer contain information that is not supported by the sources?

Return ONLY one word:

VALID

or

INVALID
"""

    response = llm.invoke(prompt)

    validation = response.content.strip().upper()

    return {
        **state,
        "validation": validation,
    }

def decide_quality(state: AgentState) -> AgentState:

    if state["retry_count"] >= 1:
        return {
            **state,
            "decision": "answer",
        }
    results = state["evaluated_results"]
    evidence = state["evidence_summary"]
    retry_count = state["retry_count"]

    # No evidence at all
    if not results or not evidence.strip():
        decision = "improve"

    else:
        average_score = sum(
            result["score"]
            for result in results
        ) / len(results)

        average_relevance = sum(
            result["relevance"]
            for result in results
        ) / len(results)

        prompt = f"""
You are a research quality gate.

USER QUESTION:
{state["user_query"]}

RESEARCH PLAN:
{state["research_plan"]}

EVALUATED SOURCES:
{results}

SYNTHESIZED EVIDENCE:
{evidence}

Average source score:
{average_score:.2f}

Average relevance:
{average_relevance:.2f}

Determine whether the research is strong enough to answer the
user's original question.

A research result is GOOD when:

- Sources are relevant to the original question.
- Multiple useful findings are supported by evidence.
- The evidence contains meaningful information rather than generic
  descriptions.
- Important research-plan aspects have reasonable evidence coverage.
- The answer can be produced without inventing information.

A research result is WEAK when:

- Most evidence focuses on only one narrow topic.
- Important aspects of the research plan have no evidence.
- Sources are only loosely related to the question.
- The evidence is too generic to identify meaningful findings.

IMPORTANT:

Do not require every research-plan aspect to be covered.

For broad questions, however, avoid accepting research that focuses
on only one narrow area when several important areas are missing.

Return ONLY:

GOOD

or

WEAK
"""

        response = llm.invoke(prompt)

        quality = response.content.strip().upper()

        if quality == "GOOD":
            decision = "answer"

        elif retry_count < 1:
            decision = "improve"

        else:
            # Maximum one retry.
            decision = "answer"

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
    evidence = state["evidence_summary"]

    prompt = f"""
You are improving the search strategy of an AI research agent.

USER QUESTION:
{state["user_query"]}

RESEARCH PLAN:
{state["research_plan"]}

PREVIOUS QUERIES:
{state["sub_queries"]}

SYNTHESIZED EVIDENCE:
{evidence}

Generate exactly 3 NEW search queries.

Your goal is to find evidence that is MISSING from the current research.

IMPORTANT:

Do NOT search again for topics that already have strong evidence.

Current evidence already covers:
- model interpretability
- data quality / data engineering

Therefore prioritize research-plan aspects that are currently missing,
especially:

- AI applications and agents
- Model and LLM engineering
- Evaluation and reliability
- Infrastructure and deployment

QUERY DESIGN:

Query 1:
Target AI applications, AI agents, or agentic systems.

Query 2:
Target LLM/model engineering, evaluation, reliability, or related
engineering practices not already covered.

Query 3:
Target AI infrastructure, deployment, scalability, observability,
or production operations.

RULES:

- Exactly 3 queries.
- Each query must contain 5 to 12 words.
- Preserve the original question's intent.
- Focus on recent/current developments.
- Make each query meaningfully different.
- Do not mention topics already sufficiently covered unless necessary.
- Do not use generic queries.
- Do not add explanations.
- Do not number the queries.

Return ONLY the three queries, one per line.
"""

    response = llm.invoke(prompt)

    lines = [
        line.strip()
        for line in response.content.splitlines()
        if line.strip()
    ]

    queries = []

    for line in lines:
        lower = line.lower()

        if lower.startswith("here are"):
            continue

        if lower.startswith("queries:"):
            continue

        if lower.startswith("query 1"):
            continue

        if lower.startswith("query 2"):
            continue

        if lower.startswith("query 3"):
            continue

        queries.append(line)

    return {
        **state,
        "sub_queries": queries[:3],
    }