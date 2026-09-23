from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


from app.agents.graph import agent


app = FastAPI(
    title="EvoSearch",
    description="Self-improving research agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research")
def research(request: ResearchRequest):

    result = agent.invoke({
        "user_query": request.query,
        "sub_queries": [],
        "search_results": [],
        "evaluated_results": [],
        "final_answer": "",
        "decision": "",
        "retry_count": 0,
        "query_history": [],
        "validation": "",
        "research_plan": [],
        "evidence_summary": "",
    })

    return {
    "query": request.query,
    "answer": result["final_answer"],
    "sources": result["evaluated_results"],
    "queries_used": result["sub_queries"],
    "retry_count": result["retry_count"],
    "query_history": result["query_history"],
    "validation": result["validation"],
    "research_plan": result["research_plan"],
    "evidence_summary": result["evidence_summary"],
    }