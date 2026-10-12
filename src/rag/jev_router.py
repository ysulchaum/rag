import os

import requests


class JevAuthenticationError(RuntimeError):
    pass


def route_query_with_jev(query: str) -> bool:
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
        "model": "jev-latest",
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

    noul_value = (
        response.json().get("answers", {}).get("needs_retrieval", {}).get("noul")
    )
    if not isinstance(noul_value, (int, float)):
        raise TypeError("Jev response did not include a needs_retrieval Noul value.")

    return noul_value >= 0.2
