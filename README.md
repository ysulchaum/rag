# RAG

This project stores document embeddings in PostgreSQL with LangChain Postgres
and pgvector, and uses DeepSeek for answer generation. Source documents are
split into 500-character chunks with 80-character overlap before indexing; up to
five matching chunks are included as context for each answer.

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

Each completed question is appended to `src/rag/log/rag_query_log.xlsx`,
including the routing result, retrieved context, answer, model names, and token
usage reported by Jev and DeepSeek. Missing provider usage is left blank.

To remove a file's indexed document, run `uv run rag --delete
src/rag/input_files/file1.txt`. The source file may already be deleted. If the
file still exists under `input_files`, the next normal startup will index it
again. To remove only some text, edit the source file and run `uv run rag`; its
existing vector record will be updated.

Use `rag.database.add_document(content)` to insert a document (it returns a
string ID) and `rag.database.search_vector_db(query)` to retrieve the closest
documents.
