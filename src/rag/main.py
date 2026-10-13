import argparse
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from .database import add_document, delete_document, initialize_database
from .graph import app as rag_graph
from .jev_router import JevAuthenticationError


def answer_question(query: str) -> str:
    result = rag_graph.invoke(
        {
            "query": query,
            "needs_retrieval": True,
            "context_docs": [],
            "generation": "",
        }
    )
    return result["generation"]


def _document_id(path: Path) -> str:
    return str(uuid5(NAMESPACE_URL, str(path.resolve())))


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
    parser.add_argument(
        "--delete",
        type=Path,
        metavar="FILE",
        help="Delete a previously indexed file by its path; the file may already be removed",
    )
    args = parser.parse_args()

    if args.delete is not None:
        delete_document(_document_id(args.delete))
        print(f"Deleted indexed document for {args.delete}, if it existed.")
        return

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
        add_document(content, document_id=_document_id(path))
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

        try:
            print(f"\n{answer_question(query)}")
        except JevAuthenticationError as error:
            parser.exit(1, f"rag: error: {error}\n")
