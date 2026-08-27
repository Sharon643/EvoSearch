from typing import TypedDict


class AgentState(TypedDict):
    user_query: str
    sub_queries: list[str]
    search_results: list[dict]
    final_answer: str