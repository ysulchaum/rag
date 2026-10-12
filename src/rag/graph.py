from typing import TypedDict

from langgraph.graph import END, StateGraph

from .config import llm_client
from .database import search_vector_db
from .jev_router import route_query_with_jev


# 1. State Definition
class RAGState(TypedDict):
    query: str
    needs_retrieval: bool
    context_docs: list[str]
    generation: str


# 2. Nodes
def classification_node(state: RAGState):
    needs_retrieval = route_query_with_jev(state["query"])
    return {"needs_retrieval": needs_retrieval}


def retrieval_node(state: RAGState):
    docs = search_vector_db(state["query"])
    return {"context_docs": docs}


def generation_node(state: RAGState):
    context = "\n".join(state.get("context_docs", []))

    prompt = f"""Answer the user's question accurately and concisely.
Use the context when provided. If the context does not contain the answer, say so.
Respond in the same language as the question.

Context:
{context or "No context retrieved."}

Question: {state["query"]}
"""
    response = llm_client.generate(prompt)
    return {"generation": response}


# 3. Conditional Routing Logic
def routing_edge(state: RAGState):
    if state["needs_retrieval"]:
        return "retrieve"
    return "generate"


# 4. Building the Graph compilation
workflow = StateGraph(RAGState)

workflow.add_node("classify", classification_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)

workflow.set_entry_point("classify")

workflow.add_conditional_edges(
    "classify",
    routing_edge,
    {
        "retrieve": "retrieve",
        "generate": "generate",
    },
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
