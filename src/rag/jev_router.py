import os
import requests
from typing import Dict, Any

def route_query_with_jev(query: str) -> Dict[str, Any]:
    """
    Uses TypeSafe Jev (System One Classifier) to determine if a query requires database retrieval.
    """
    api_key = os.getenv("TYPESAFE_API_KEY")
    url = "https://api.typesafe-ai"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Passing the text query as input state alongside a typed Noul (Boolean primitive)
    payload = {
        "model": "jev-latest",
        "state": query,
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": "Is this request urgent?"
            }
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    
    # Returns structured payload like {"needs_retrieval": {"value": True, "confidence": 0.94}}
    return response.json().get("results", {})
