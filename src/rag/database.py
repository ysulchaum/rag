from uuid import NAMESPACE_URL, uuid4, uuid5

from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import embed_text, settings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
_text_splitter = RecursiveCharacterTextSplitter(
	chunk_size=CHUNK_SIZE,
	chunk_overlap=CHUNK_OVERLAP,
)


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


def add_document(content: str, document_id: str | None = None) -> str:
	if not content.strip():
		raise ValueError("Document content cannot be empty.")

	source_id = document_id or str(uuid4())
	chunks = _text_splitter.split_text(content)
	chunk_ids = [
		str(uuid5(NAMESPACE_URL, f"{source_id}:chunk:{index}"))
		for index in range(len(chunks))
	]
	metadata = [{"source_id": source_id} for _ in chunks]

	delete_document(source_id)
	_get_vector_store().add_texts(chunks, metadatas=metadata, ids=chunk_ids)
	return source_id


def delete_document(document_id: str) -> None:
	vector_store = _get_vector_store()
	with vector_store._make_sync_session() as session:
		collection = vector_store.get_collection(session)
		chunk_ids: list[str] = []
		if collection is not None:
			embedding_store = vector_store.EmbeddingStore
			rows = (
				session.query(embedding_store.id)
				.filter(
					embedding_store.collection_id == collection.uuid,
					embedding_store.cmetadata["source_id"].as_string() == document_id,
				)
				.all()
			)
			chunk_ids = [str(row[0]) for row in rows]

	vector_store.delete(ids=[document_id, *chunk_ids], collection_only=True)


def search_vector_db(query: str, limit: int = 5) -> list[str]:
	if limit < 1:
		raise ValueError("Search limit must be at least 1.")

	documents = _get_vector_store().similarity_search(query, k=limit)
	return [document.page_content for document in documents]
