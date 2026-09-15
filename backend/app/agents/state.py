from typing import TypedDict


class AgentState(TypedDict):
    user_query: str
    sub_queries: list[str]
    search_results: list[dict]
    evaluated_results: list[dict]
    final_answer: str
    decision: str
    retry_count: int
    query_history: list[str]
    validation: str
    research_plan: list[str]
    evidence_summary: str