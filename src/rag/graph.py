from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from src.rag.jev_router import route_query_with_jev
from src.rag.database import search_vector_db
from src.rag.config import llm_client

# 1. State Definition
class RAGState(TypedDict):
    query: str
    needs_retrieval: bool
    context_docs: List[str]
    generation: str

# 2. Nodes
def classification_node(state: RAGState):
    jev_result = route_query_with_jev(state["query"])
    # Pull the Boolean value out of Jev's strict type response
    needs_retrieval = jev_result.get("needs_retrieval", {}).get("value", True)
    return {"needs_retrieval": needs_retrieval}

def retrieval_node(state: RAGState):
    docs = search_vector_db(state["query"])
    return {"context_docs": docs}

def generation_node(state: RAGState):
    context = "\n".join(state.get("context_docs", []))
    
    prompt = f"""
    Context Info: {context if context else 'No context retrieved.'}
    User Query: {state['query']}
    
    Provide an accurate, concise answer based on the context information above.
    """
    # Standard LLM execution string text generation
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
        "generate": "generate"
    }
)
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()
