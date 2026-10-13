import os
from dataclasses import dataclass

import requests

JEV_MODEL = "jev-latest"


class JevAuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class JevRouteResult:
    needs_retrieval: bool
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    model: str


def _token_count(usage: object, name: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(name)
    if type(value) is int and value >= 0:
        return value
    return None


def classify_query_with_jev(query: str) -> JevRouteResult:
    """
    Uses TypeSafe Jev (System One Classifier) to determine if a query requires database retrieval.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    if not api_key:
        raise JevAuthenticationError(
            "Set TYPESAFE_API_KEY to use Jev query classification."
        )

    url = "https://api.typesafe.ai/v1/systemone"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Passing the text query as input state alongside a typed Noul (Boolean primitive)
    payload = {
        "model": JEV_MODEL,
        "state": query,
        "questions": {
            "needs_retrieval": {
                "type": "noul",
                "instructions": "Is the user asking about, naming, or seeking information on a topic that may be covered by the indexed documents? Treat short topic phrases, keywords, and section headings as yes. Answer no only for greetings or requests clearly unrelated to the documents.",
                "criteria": {
                    "true": "The request asks about any topic, keyword, or heading that could be covered by the indexed documents, including broad or incomplete phrases.",
                    "false": "The request is only a greeting, thanks, or clearly unrelated to the indexed documents.",
                },
            }
        },
    }

    response = requests.post(url, json=payload, headers=headers, timeout=30)
    if response.status_code == 401:
        raise JevAuthenticationError(
            "TypeSafe rejected TYPESAFE_API_KEY (HTTP 401). Create a valid key in the TypeSafe Console and replace it in rag-server/.env."
        )
    response.raise_for_status()

    response_data = response.json()
    noul_value = (
        response_data.get("answers", {}).get("needs_retrieval", {}).get("noul")
    )
    if not isinstance(noul_value, (int, float)):
        raise TypeError("Jev response did not include a needs_retrieval Noul value.")

    usage = response_data.get("usage")
    return JevRouteResult(
        needs_retrieval=noul_value >= 0.2,
        input_tokens=_token_count(usage, "input_tokens"),
        output_tokens=_token_count(usage, "output_tokens"),
        total_tokens=_token_count(usage, "total_tokens"),
        model=JEV_MODEL,
    )


def route_query_with_jev(query: str) -> bool:
    return classify_query_with_jev(query).needs_retrieval
