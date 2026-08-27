import os

from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)


def search_web(query: str) -> list[dict]:
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=False,
    )

    return response.get("results", [])