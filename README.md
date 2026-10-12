# RAG

This project stores document embeddings in PostgreSQL with LangChain Postgres
and pgvector, and uses DeepSeek for answer generation.

Start PostgreSQL and Ollama with `docker compose up -d`. Download the embedding
model into the Ollama container once with `docker compose exec ollama ollama pull
bge-m3`. The model is persisted in a Docker volume. The default database URL
matches the credentials in `docker-compose.yml`; override it with
`DATABASE_URL` when needed.

Set `DEEPSEEK_API_KEY` for chat generation. Embeddings run locally through
Ollama with the `bge-m3` model. Change the Ollama endpoint or model with
`EMBEDDING_BASE_URL` or `EMBEDDING_MODEL`.

Use `rag.database.add_document(content)` to insert a document (it returns a
string ID) and `rag.database.search_vector_db(query)` to retrieve the closest
documents.
