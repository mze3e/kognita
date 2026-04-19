# Local embedding server

Tiny OpenAI-compatible `/v1/embeddings` server backed by
[sentence-transformers](https://www.sbert.net/). Lets you run the Streamlit
demo (or the `kognita` library) fully offline without paying for OpenAI
embeddings.

## Run

From the repo root:

```bash
pip install -e ".[local-embeddings]"
uvicorn examples.local_embedding_server.server:app --host 127.0.0.1 --port 8000
```

Health check: `curl http://127.0.0.1:8000/health`.

## Configuration

- `LOCAL_EMBEDDING_MODEL` — HuggingFace model ID (default `BAAI/bge-small-en-v1.5`).
- `LOCAL_EMBEDDING_NAME` — short name returned by `/v1/models` (default derived from the model ID).

## Use with `kognita`

```python
from kognita import EmbedderConfig

embedder = EmbedderConfig(
    provider="local",
    model="bge-small-en-v1.5",
    dimension=384,
    api_key="dummy",
    base_url="http://127.0.0.1:8000/v1",
)
```
