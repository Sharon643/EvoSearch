from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes import (
    analyze_query,
    search_sources,
    evaluate_sources,
    generate_answer,
    decide_quality,
    improve_query,
)


def quality_router(state: AgentState):
    return state["decision"]


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search_sources", search_sources)
    graph.add_node("evaluate_sources", evaluate_sources)
    graph.add_node("decide_quality", decide_quality)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("improve_query", improve_query)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search_sources")
    graph.add_edge("search_sources", "evaluate_sources")
    graph.add_edge("evaluate_sources", "decide_quality")

    graph.add_conditional_edges(
        "decide_quality",
        quality_router,
        {
            "answer": "generate_answer",
            "improve": "improve_query",
        },
    )

    graph.add_edge("improve_query", "search_sources")
    graph.add_edge("generate_answer", END)

    return graph.compile()


agent = build_graph()