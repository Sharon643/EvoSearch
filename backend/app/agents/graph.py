from langgraph.graph import StateGraph, START, END

from app.agents.state import AgentState
from app.agents.nodes import (
    analyze_query,
    search_sources,
    generate_answer,
)


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("analyze_query", analyze_query)
    graph.add_node("search_sources", search_sources)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "analyze_query")
    graph.add_edge("analyze_query", "search_sources")
    graph.add_edge("search_sources", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


agent = build_graph()