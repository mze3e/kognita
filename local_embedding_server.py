"""
OpenAI-compatible local embedding server for Kognita.

Run with:
    uvicorn local_embedding_server:app --host 127.0.0.1 --port 8000

Optional env vars:
    LOCAL_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
    LOCAL_EMBEDDING_NAME=bge-small-en-v1.5
"""
import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


MODEL_ID = os.environ.get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
MODEL_NAME = os.environ.get("LOCAL_EMBEDDING_NAME", MODEL_ID.rsplit("/", 1)[-1])

app = FastAPI(title="Kognita Local Embeddings", version="0.1.0")
model = SentenceTransformer(MODEL_ID)


class EmbeddingRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    input: str | list[str]
    encoding_format: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "embedding_dim": model.get_sentence_embedding_dimension(),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest) -> dict[str, Any]:
    texts = req.input if isinstance(req.input, list) else [req.input]
    vectors = model.encode(texts, normalize_embeddings=True).tolist()

    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector,
            }
            for index, vector in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }
