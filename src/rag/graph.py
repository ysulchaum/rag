from typing import NotRequired, TypedDict

from langgraph.graph import END, StateGraph

from .config import llm_client, settings
from .database import search_vector_db
from .jev_router import classify_query_with_jev


# 1. State Definition
class RAGState(TypedDict):
    query: str
    needs_retrieval: bool
    context_docs: list[str]
    generation: str
    jev_model: NotRequired[str]
    jev_input_tokens: NotRequired[int | None]
    jev_output_tokens: NotRequired[int | None]
    jev_total_tokens: NotRequired[int | None]
    llm_model: NotRequired[str]
    llm_prompt_tokens: NotRequired[int | None]
    llm_completion_tokens: NotRequired[int | None]
    llm_total_tokens: NotRequired[int | None]


# 2. Nodes
def classification_node(state: RAGState):
    result = classify_query_with_jev(state["query"])
    return {
        "needs_retrieval": result.needs_retrieval,
        "jev_model": result.model,
        "jev_input_tokens": result.input_tokens,
        "jev_output_tokens": result.output_tokens,
        "jev_total_tokens": result.total_tokens,
    }


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
    response, usage = llm_client.generate_with_usage(prompt)
    return {
        "generation": response,
        "llm_model": settings.deepseek_model,
        "llm_prompt_tokens": usage.input_tokens,
        "llm_completion_tokens": usage.output_tokens,
        "llm_total_tokens": usage.total_tokens,
    }


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
