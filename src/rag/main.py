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
    default_source = Path(__file__).parent / "input_files"
    parser = argparse.ArgumentParser(
        description="Ingest text documents and ask questions about them."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=default_source,
        help="UTF-8 text/Markdown file or directory (default: input_files)",
    )
    args = parser.parse_args()

    if args.source.is_file():
        documents = [args.source]
    elif args.source.is_dir():
        documents = sorted(
            (
                path
                for path in args.source.rglob("*")
                if path.is_file() and path.suffix.lower() in {".txt", ".md"}
            ),
            key=lambda path: str(path).casefold(),
        )
        if not documents:
            parser.error(f"No .txt or .md files found in {args.source}")
    else:
        parser.error(f"Source not found: {args.source}")

    document_contents = [(path, path.read_text(encoding="utf-8")) for path in documents]
    initialize_database()
    for path, content in document_contents:
        document_id = str(uuid5(NAMESPACE_URL, str(path.resolve())))
        add_document(content, document_id=document_id)
    print(
        f"Indexed {len(documents)} files in pgvector. Ask a question, or type 'exit' to quit."
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
