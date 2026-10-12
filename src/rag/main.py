import argparse
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .config import llm_client
from .database import add_document, initialize_database, search_vector_db


def answer_question(query: str) -> str:
    context_documents = search_vector_db(query)
    context = "\n\n".join(context_documents)
    prompt = f"""Answer the user's question using the context below.
If the context does not contain the answer, say that you do not know based on the provided document.
Respond in the same language as the question.

Context:
{context or "No relevant context was found."}

Question: {query}
"""
    return llm_client.generate(prompt)


def main() -> None:
    default_document = Path(__file__).parent / "input_files" / "demo.txt"
    parser = argparse.ArgumentParser(
        description="Ingest a document and ask questions about it."
    )
    parser.add_argument(
        "document",
        nargs="?",
        type=Path,
        default=default_document,
        help=f"UTF-8 text file to ingest (default: {default_document.name})",
    )
    args = parser.parse_args()

    if not args.document.is_file():
        parser.error(f"Document not found: {args.document}")

    content = args.document.read_text(encoding="utf-8")
    initialize_database()
    document_id = str(uuid5(NAMESPACE_URL, str(args.document.resolve())))
    add_document(content, document_id=document_id)
    print(
        f"Indexed {args.document} in pgvector. Ask a question, or type 'exit' to quit."
    )

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except EOFError:
            print()
            break

        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        print(f"\n{answer_question(query)}")
