import requests


SEARXNG_URL = "http://localhost:8080/search"


def search_web(query: str) -> list[dict]:
    response = requests.get(
        SEARXNG_URL,
        params={
            "q": query,
            "format": "json",
        },
        timeout=10,
    )

    response.raise_for_status()

    data = response.json()

    return [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
        }
        for result in data.get("results", [])
    ]