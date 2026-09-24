# EvoSearch

An AI-powered research agent that searches the web, evaluates evidence, synthesizes findings, and validates its answers.

EvoSearch is designed to go beyond the traditional:

Question → LLM → Answer

approach.

Instead, it uses a multi-step research workflow that plans the research, retrieves sources, evaluates them, synthesizes evidence, checks research quality, and validates the final response.

---

## Overview

EvoSearch takes a natural-language research question and performs the following workflow:

```text
User Question
      │
      ▼
Analyze Query
      │
      ▼
Generate Research Queries
      │
      ▼
Search Web
      │
      ▼
Evaluate Sources
      │
      ▼
Synthesize Evidence
      │
      ▼
Quality Gate
      │
      ├── Good ──────────────┐
      │                      ▼
      │                 Generate Answer
      │                      │
      │                      ▼
      │                 Validate Answer
      │                      │
      │                      ▼
      │                    Result
      │
      └── Weak
             │
             ▼
        Improve Queries
             │
             ▼
        Search Again

The system can perform one additional research cycle when the initial evidence is considered insufficient.

Features
AI-powered research planning
Automatic query decomposition
Multiple web searches per research task
SearXNG integration for web search
Source deduplication
Source relevance evaluation
Source authority evaluation
Source freshness evaluation
Evidence synthesis
Research quality gate
Automatic query improvement
Answer generation from synthesized evidence
Final answer validation
React-based research interface
Displays retrieved sources and research queries
Tech Stack
Backend
Python
FastAPI
LangGraph
LangChain
Ollama
SearXNG
Frontend
React
Vite
Axios
CSS
Infrastructure
Docker
Docker Compose
SearXNG
Ollama
Architecture

The backend is implemented as a LangGraph workflow.

Agent State

The agent maintains information such as:

User query
Research plan
Search queries
Search results
Evaluated sources
Evidence summary
Retry count
Query history
Generated answer
Validation result
Graph
START
  │
  ▼
Analyze Query
  │
  ▼
Search Sources
  │
  ▼
Evaluate Sources
  │
  ▼
Synthesize Evidence
  │
  ▼
Quality Gate
  │
  ├─────────────── Good ───────────────► Generate Answer
  │                                           │
  │                                           ▼
  │                                     Validate Answer
  │                                           │
  │                                           ▼
  │                                          END
  │
  └─────────────── Weak
                      │
                      ▼
                 Improve Query
                      │
                      ▼
                 Search Sources
Project Structure
EvoSearch/
│
├── backend/
│   ├── app/
│   │   ├── agents/
│   │   │   ├── graph.py
│   │   │   ├── nodes.py
│   │   │   └── state.py
│   │   │
│   │   ├── tools/
│   │   │   └── search.py
│   │   │
│   │   └── main.py
│   │
│   ├── .env
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   │
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── searxng/
│
├── .gitignore
└── README.md
How It Works
1. Query Analysis

The user's question is analyzed to identify the important research dimensions.

The agent then generates three focused search queries.

For example:

User:
What are the latest AI engineering trends?

The agent may break this into areas such as:

AI applications and agents
Model and LLM engineering
Evaluation and reliability
Infrastructure and deployment
2. Web Search

The generated queries are sent to SearXNG.

Each query retrieves multiple web results.

EvoSearch then removes duplicate URLs before continuing with the research pipeline.

3. Source Evaluation

Retrieved sources are evaluated using three dimensions:

Relevance
Authority
Freshness

These scores are combined into an overall source score.

This allows the research pipeline to prioritize sources that are more useful for the original question.

4. Evidence Synthesis

The evaluated sources are passed to the synthesis stage.

The synthesis component extracts meaningful findings from the available evidence instead of simply summarizing individual search results.

Each finding is associated with the sources supporting it.

5. Quality Gate

The research is then evaluated by a quality gate.

The system checks whether:

The sources are relevant.
The evidence contains useful findings.
The research covers important aspects of the question.
The available evidence is sufficient to generate an answer.

If the research is weak, EvoSearch generates improved search queries.

6. Query Improvement

When the initial research is insufficient, the agent identifies missing areas and generates new search queries targeting those gaps.

The system performs a maximum of one retry to prevent unnecessary search loops.

7. Answer Generation

The final answer is generated using the synthesized evidence.

The answer is instructed to:

Stay within the available evidence.
Avoid unsupported claims.
Include source references.
Avoid exaggerating the research findings.
8. Answer Validation

Before returning the result, EvoSearch validates the generated answer against the retrieved sources.

The validation stage checks whether the answer contains unsupported claims or incorrect source references.

Frontend

The React frontend provides a simple research interface.

Users can:

Enter a research question.
Start the research process.
View the generated answer.
View the retrieved sources.
Inspect source evaluation metrics.
View the queries used by the research agent.
See the research stages while the agent is running.
Example
Input
What are the latest AI engineering trends?
Research Process
Analyze question
       ↓
Generate research queries
       ↓
Search web
       ↓
Evaluate sources
       ↓
Synthesize evidence
       ↓
Quality check
       ↓
Generate answer
       ↓
Validate answer
Output

The interface displays:

Research answer
Retrieved sources
Source metrics
Search queries
Validation status
Number of retries
Local Development
Requirements

Install the following:

Python 3.11+
Node.js
Docker Desktop
Ollama
1. Start SearXNG

Navigate to the SearXNG directory:

cd searxng

Start the container:

docker compose up -d

SearXNG should then be available at:

http://localhost:8080
2. Start Ollama

Make sure Ollama is running and the required model is available.

Check installed models:

ollama list

For example:

ollama run llama3
3. Start the Backend

Navigate to the backend:

cd backend

Create the Python virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Start FastAPI:

uvicorn app.main:app --port 8000

The backend will run at:

http://localhost:8000

FastAPI documentation:

http://localhost:8000/docs
4. Start the Frontend

Open another terminal and navigate to:

cd frontend

Install dependencies:

npm install

Start the development server:

npm run dev

Open the URL displayed by Vite in your browser.

API
Health Check
GET /health

Example response:

{
  "status": "ok"
}
Research
POST /research

Request:

{
  "query": "What are the latest AI engineering trends?"
}

The response contains:

Generated answer
Evaluated sources
Queries used
Research plan
Retry count
Query history
Validation status
Design Goals

EvoSearch was built around several AI engineering principles:

Agentic Workflows

Instead of a single LLM call, the system uses multiple specialized stages connected through a stateful graph.

Evidence-Based Generation

The answer generation stage receives synthesized evidence rather than relying solely on the model's internal knowledge.

Self-Improvement

The system can identify weak research and generate improved queries for another research cycle.

Validation

The final answer is checked against the available evidence before being returned.

Observability

The frontend exposes the research process, queries, sources, and validation status so users can understand how the answer was produced.

Future Improvements

Potential future improvements include:

Streaming agent execution
Parallel search execution
Persistent research history
Search result caching
More robust structured LLM outputs
Multi-model support
Advanced source ranking
Research memory
Production deployment
Authentication
Background research jobs