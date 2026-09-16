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

Create a research plan that identifies the important dimensions
needed to answer the user's question comprehensively.

For questions asking about "latest trends", "current trends",
"emerging trends", or similar broad topics:

- Cover multiple distinct dimensions of the field.
- Do not focus the entire plan on one technology or technique.
- Consider areas such as:
  AI applications and agents,
  model and LLM engineering,
  evaluation and reliability,
  data and context engineering,
  infrastructure and deployment,
  AI-assisted software engineering,
  observability and operations.
- Only include dimensions that are actually relevant to the question.
- Do not assume all of these areas must be included.

For other questions:
- Identify 3 to 5 distinct aspects specifically relevant to the question.
- Do not introduce unrelated domains.

Then generate exactly 3 search queries.

Each query should investigate a DIFFERENT aspect of the research plan.

Rules for queries:
- Keep each query between 5 and 12 words.
- Preserve the user's original intent.
- Make queries specific enough to produce useful sources.
- Avoid overlapping queries.
- For "latest/current/recent" questions, explicitly target recent information.
- Do not include unnecessary phrases such as:
  "with a focus on",
  "according to leading experts",
  "research papers and industry reports".
- Do not add years that are older than the current year.
- Do not invent a specific year unless useful for the search.

Return ONLY this format:

PLAN:
aspect 1
aspect 2
aspect 3
aspect 4

QUERIES:
query 1
query 2
query 3

Do not add explanations.
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
            research_plan.append(line.rstrip(","))

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

Your job is to turn evaluated search results into reliable evidence
for answering the user's ORIGINAL question.

USER QUESTION:
{state["user_query"]}

RESEARCH PLAN:
{state["research_plan"]}

EVALUATED SOURCES:
{sources}

Follow these steps internally:

1. Examine each research-plan aspect separately.

2. Determine which sources actually provide evidence for that aspect.

3. Ignore sources that are only loosely related to the aspect.

4. For questions about trends, identify actual trends, developments,
   engineering practices, technologies, or changes in the field.

5. Do NOT treat an isolated application example, statistic, product,
   or use case as a broad industry trend unless the sources explicitly
   support that interpretation.

6. Combine overlapping findings from multiple sources.

7. Prefer findings supported by multiple independent sources.

8. A single highly authoritative source may support a finding, but
   mark its confidence as MEDIUM unless the evidence is especially
   direct and strong.

9. Do not invent information to fill missing research-plan aspects.

10. If sources do not provide sufficient evidence for an aspect,
    explicitly mark that aspect as having insufficient evidence.

11. Do not force every source into the final synthesis.

12. Do not use information that is not present in the sources.

For each supported finding, use exactly this structure:

THEME:
Short name of the trend or finding.

RESEARCH PLAN ASPECT:
The research-plan aspect this finding belongs to.

FINDING:
Concise explanation of what the sources actually support.

SOURCES:
Source numbers supporting the finding.

CONFIDENCE:
HIGH, MEDIUM, or LOW

After the findings, provide:

COVERAGE:
For each research-plan aspect, state:
SUPPORTED or INSUFFICIENT EVIDENCE

IMPORTANT:
Do not call something a "latest trend" simply because it appears
in a recent article.

The finding must represent a meaningful trend, development, practice,
or change relevant to the original question.

Return only the synthesized evidence.
Do not add general knowledge.
Do not add recommendations.
Do not add information outside the sources.
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
- Answer the user's question directly.
- Use only claims supported by the synthesized evidence.
- Do not invent facts.
- Preserve the confidence level indicated by the synthesis.
- Every major factual claim must include source numbers like [Source 1].
- If multiple sources support a claim, cite all relevant sources.
- Do not introduce findings that are absent from the synthesized evidence.
- If an important aspect of the research plan lacks evidence, say so.
- Do not present a single trend as the only trend when multiple supported
  trends are identified.
- If the evidence is insufficient to answer the question completely,
  explicitly say so.

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
    results = state["evaluated_results"]
    evidence = state["evidence_summary"]
    retry_count = state["retry_count"]

    if not results or not evidence.strip():
        decision = "improve"

    else:
        # Calculate source quality
        average_score = sum(
            result["score"]
            for result in results
        ) / len(results)

        average_relevance = sum(
            result["relevance"]
            for result in results
        ) / len(results)

        # Check how much of the research plan has evidence
        research_plan = state["research_plan"]

        covered_aspects = 0

        for aspect in research_plan:
            if aspect.lower() in evidence.lower():
                covered_aspects += 1

        coverage_ratio = (
            covered_aspects / len(research_plan)
            if research_plan
            else 0
        )

        # Quality decision
        if (
            average_score >= 7
            and average_relevance >= 7
            and coverage_ratio >= 0.6
        ):
            decision = "answer"

        elif retry_count >= 1:
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
Query: {result["query"]}
"""
        for result in results
    )

    prompt = f"""
You are improving a web search strategy.

Original user question:
{state["user_query"]}

Research plan:
{state["research_plan"]}

Previous search queries:
{state["sub_queries"]}

Previous evaluated results:
{sources}

Generate exactly 3 NEW search queries for the original question.

Rules:
- Keep each query short: 5 to 12 words.
- Preserve the original topic and intent.
- Do not add unrelated domains.
- Do not add phrases like "with a focus on".
- Do not mention research papers, case studies, or expert opinions
  unless they are necessary to answer the original question.
- Make the 3 queries cover different aspects of the question.
- Focus on information that was missing or weak in the previous results.
- The improved queries MUST target aspects from the research plan.
- Do not replace the research plan with new topics.
- Prioritize research-plan aspects that have little or no source coverage.
- Prefer recent information when the original question asks for latest,
  current, or recent information.
- Do not simply rewrite the previous queries.

Return ONLY exactly 3 queries.
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