from langchain_postgres.vectorstores import PGVector

from .config import embed_text, settings


class ConfiguredEmbeddings:
	def embed_documents(self, texts: list[str]) -> list[list[float]]:
		return [embed_text(text) for text in texts]

	def embed_query(self, text: str) -> list[float]:
		return embed_text(text)


_vector_store: PGVector | None = None


def _get_vector_store() -> PGVector:
	global _vector_store
	if _vector_store is None:
		connection = settings.database_url
		if connection.startswith("postgresql://"):
			connection = connection.replace("postgresql://", "postgresql+psycopg://", 1)
		_vector_store = PGVector(
			connection=connection,
			embeddings=ConfiguredEmbeddings(),
			collection_name="rag_documents",
			use_jsonb=True,
		)
	return _vector_store


def initialize_database() -> None:
	_get_vector_store()


def add_document(content: str) -> str:
	if not content.strip():
		raise ValueError("Document content cannot be empty.")

	document_ids = _get_vector_store().add_texts([content])
	return document_ids[0]


def search_vector_db(query: str, limit: int = 5) -> list[str]:
	if limit < 1:
		raise ValueError("Search limit must be at least 1.")

	documents = _get_vector_store().similarity_search(query, k=limit)
	return [document.page_content for document in documents]
