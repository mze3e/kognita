# PDF → Knowledge Graph (Graphiti + Streamlit)

Turn any PDF into an interactive, queryable knowledge graph using **Graphiti** (by Zep) as the graph engine, **KuzuDB** as the embedded graph database, and **Streamlit** for the UI.

## How it works

1. **Upload** a PDF → text is extracted with PyMuPDF
2. **Chunked** into overlapping word windows (configurable)
3. Each chunk is fed to **Graphiti** as a `text` episode
4. Graphiti calls **Claude (Anthropic)** to extract entities & relationships
5. Relationships are embedded via **OpenAI** (`text-embedding-3-small`)
6. The resulting graph is stored in **KuzuDB** (embedded, no Docker needed)
7. **Pyvis** renders the interactive graph in the browser
8. You can **search** the graph with natural language queries

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API Keys needed

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | Entity & relationship extraction (Claude Sonnet) |
| `OPENAI_API_KEY` | Text embeddings (`text-embedding-3-small`) |

You can set these as env vars or enter them in the sidebar at runtime.

## Tips

- **Chunk size**: 200–300 words works well for most documents
- **Overlap**: 30 words helps preserve context at chunk boundaries
- Each episode costs ~2–5 LLM calls (extraction + resolution + dedup)
- For large PDFs (50+ pages), expect 2–5 min processing time
- Use the **Export** button to download the graph as JSON for further analysis
