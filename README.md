# RAG

This project stores document embeddings in PostgreSQL with LangChain Postgres
and pgvector, and uses DeepSeek for answer generation.

Start PostgreSQL and Ollama with `docker compose up -d`. Download the embedding
model into the Ollama container once with `docker compose exec ollama ollama pull
bge-m3`. The model is persisted in a Docker volume. The default database URL
matches the credentials in `docker-compose.yml`; override it with
`DATABASE_URL` when needed.

Set `DEEPSEEK_API_KEY` for chat generation and `TYPESAFE_API_KEY` for Jev query
classification. LangGraph runs the question flow: Jev decides whether retrieval
is needed, pgvector supplies matching document context when needed, and DeepSeek
generates the answer. Embeddings run locally through Ollama with the `bge-m3`
model. Change the Ollama endpoint or model with `EMBEDDING_BASE_URL` or
`EMBEDDING_MODEL`.

Run `uv run rag` from this directory to initialize pgvector, embed all `.txt`
and `.md` files under `src/rag/input_files`, and start an interactive question
prompt. Files are re-indexed with the same IDs on each run. Type `exit` or
`quit` to leave. Pass a UTF-8 text file or directory to ingest a different
source, such as `uv run rag path/to/documents`.

Use `rag.database.add_document(content)` to insert a document (it returns a
string ID) and `rag.database.search_vector_db(query)` to retrieve the closest
documents.
