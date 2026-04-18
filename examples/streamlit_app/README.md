# Streamlit demo

Interactive PDF → knowledge graph UI built on top of the `kognita` library.

This example is the original Kognita UI, relocated here unchanged. It still
bundles its own graph-building logic; a follow-up change will refactor it to
call `kognita.Kognita(...)` directly.

## Run

From the repo root:

```bash
pip install -e ".[demo]"
streamlit run examples/streamlit_app/app.py
```

Saved graphs are written to `.saved_graphs/` in whatever directory you run the
command from, so keep your working directory consistent across sessions.

## Environment variables

Configure at least one LLM provider and the embedder via `.env` or your shell:

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`
- `OLLAMA_BASE_URL`, `OLLAMA_LLM_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_EMBED_DIM`
- `LOCAL_EMBEDDINGS_BASE_URL`, `LOCAL_EMBEDDINGS_MODEL`, `LOCAL_EMBEDDINGS_DIM`
- `CUSTOM_OPENAI_BASE_URL`, `CUSTOM_OPENAI_API_KEY`, `CUSTOM_OPENAI_EMBED_MODEL`, `CUSTOM_OPENAI_EMBED_DIM`

See the top-level `README.md` for the full list and feature walkthrough.
