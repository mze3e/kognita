"""
PDF → Knowledge Graph  (Graphiti + KuzuDB + Streamlit)
-------------------------------------------------------
Upload a PDF → chunked into episodes → Graphiti extracts entities &
relationships via Claude Sonnet → interactive pyvis graph + Q&A search.
"""
import asyncio
import json
import os
import hashlib
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
import requests

import fitz  # PyMuPDF
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import pandas as pd
import openai
import kuzu
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ── Graphiti ──────────────────────────────────────────────────────────────────
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.graph_queries import get_fulltext_indices
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.nodes import EntityNode, EpisodeType

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
load_dotenv()

SAVED_GRAPHS_DIR = ".saved_graphs"
os.makedirs(SAVED_GRAPHS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Utility functions for saving/loading graphs
# ═══════════════════════════════════════════════════════════════════════════════

def get_pdf_hash(pdf_bytes: bytes) -> str:
    """Generate a hash for the PDF content."""
    return hashlib.md5(pdf_bytes).hexdigest()

def get_saved_graphs() -> list[dict]:
    """Get list of saved graphs with metadata."""
    saved_graphs = []
    if os.path.exists(SAVED_GRAPHS_DIR):
        for dirname in os.listdir(SAVED_GRAPHS_DIR):
            graph_dir = os.path.join(SAVED_GRAPHS_DIR, dirname)
            if os.path.isdir(graph_dir):
                metadata_file = os.path.join(graph_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        saved_graphs.append(metadata)
                    except Exception:
                        continue
    return sorted(saved_graphs, key=lambda x: x.get('processed_at', ''), reverse=True)

def save_graph_data(pdf_name: str, pdf_bytes: bytes, nodes: dict, edges: list, episodes: list, processing_model: str):
    """Save graph data to disk."""
    pdf_hash = get_pdf_hash(pdf_bytes)
    graph_dir = os.path.join(SAVED_GRAPHS_DIR, f"{pdf_hash}_{int(datetime.now().timestamp())}")

    os.makedirs(graph_dir, exist_ok=True)

    # Save PDF
    pdf_path = os.path.join(graph_dir, f"{pdf_name}")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)

    # Save graph data
    graph_data = {
        "nodes": [
            {
                "uuid": n.uuid,
                "name": n.name,
                "summary": n.summary,
                "labels": n.labels,
            }
            for n in nodes.values()
        ],
        "edges": [
            {
                "uuid": e.uuid,
                "source_node_uuid": e.source_node_uuid,
                "target_node_uuid": e.target_node_uuid,
                "fact": e.fact,
                "name": e.name,
            }
            for e in edges
        ],
        "episodes": episodes,
    }

    with open(os.path.join(graph_dir, "graph_data.json"), 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)

    # Save metadata
    metadata = {
        "pdf_name": pdf_name,
        "pdf_hash": pdf_hash,
        "processed_at": datetime.now().isoformat(),
        "processing_model": processing_model,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "episode_count": len(episodes),
        "graph_dir": graph_dir,
    }

    with open(os.path.join(graph_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    return graph_dir

def load_graph_data(graph_dir: str) -> tuple[dict, list, list, bytes, str]:
    """Load graph data from disk."""
    # Load graph data
    with open(os.path.join(graph_dir, "graph_data.json"), 'r') as f:
        graph_data = json.load(f)

    # Load metadata
    with open(os.path.join(graph_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)

    # Load PDF
    pdf_files = [f for f in os.listdir(graph_dir) if f.endswith('.pdf')]
    if pdf_files:
        with open(os.path.join(graph_dir, pdf_files[0]), 'rb') as f:
            pdf_bytes = f.read()
    else:
        pdf_bytes = b""

    # Convert back to proper objects
    nodes = {}
    for node_data in graph_data["nodes"]:
        # Create a simple dict representation that matches what we need
        node = {
            "uuid": node_data["uuid"],
            "name": node_data["name"],
            "summary": node_data["summary"],
            "labels": node_data["labels"],
        }
        nodes[node["uuid"]] = type('EntityNode', (), node)()

    edges = []
    for edge_data in graph_data["edges"]:
        edge = type('EntityEdge', (), edge_data)()
        edges.append(edge)

    return nodes, edges, graph_data["episodes"], pdf_bytes, metadata["pdf_name"]

def is_pdf_already_processed(pdf_bytes: bytes) -> str:
    """Check if PDF has already been processed and return graph_dir if found."""
    pdf_hash = get_pdf_hash(pdf_bytes)
    if os.path.exists(SAVED_GRAPHS_DIR):
        for dirname in os.listdir(SAVED_GRAPHS_DIR):
            graph_dir = os.path.join(SAVED_GRAPHS_DIR, dirname)
            if os.path.isdir(graph_dir):
                metadata_file = os.path.join(graph_dir, "metadata.json")
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        if metadata.get("pdf_hash") == pdf_hash:
                            return graph_dir
                    except Exception:
                        continue
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# Model fetching functions
# ═══════════════════════════════════════════════════════════════════════════════

def get_anthropic_models(api_key: str) -> list[str]:
    """Fetch available models from Anthropic API."""
    if not api_key:
        return []
    try:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        response = requests.get("https://api.anthropic.com/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            print(f"Failed to fetch Anthropic models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Anthropic models: {e}")
        return []

def get_openai_models(api_key: str) -> list[str]:
    """Fetch available models from OpenAI API."""
    if not api_key:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Filter for chat models
            chat_models = [model["id"] for model in data.get("data", []) if model["id"].startswith(("gpt-", "chatgpt-"))]
            return sorted(chat_models)
        else:
            print(f"Failed to fetch OpenAI models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching OpenAI models: {e}")
        return []

def get_groq_models(api_key: str) -> list[str]:
    """Fetch available models from Groq API (OpenAI-compatible)."""
    if not api_key:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        else:
            print(f"Failed to fetch Groq models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Groq models: {e}")
        return []

def get_gemini_models(api_key: str) -> list[str]:
    """Fetch available models from Google Gemini API."""
    if not api_key:
        return []
    try:
        headers = {"x-goog-api-key": api_key}
        response = requests.get("https://generativelanguage.googleapis.com/v1beta/models", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["name"].replace("models/", "") for model in data.get("models", []) if "generateContent" in model.get("supportedGenerationMethods", [])]
        else:
            print(f"Failed to fetch Gemini models: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching Gemini models: {e}")
        return []

def get_ollama_models(base_url: str) -> list[str]:
    """Fetch models from a running Ollama instance via its OpenAI-compatible endpoint."""
    if not base_url:
        return []
    try:
        url = get_openai_compatible_base_url(base_url) + "/models"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return [m["id"] for m in response.json().get("data", [])]
        return []
    except Exception:
        return []


def get_custom_models(base_url: str, api_key: str) -> list[str]:
    """Fetch models from any OpenAI-compatible endpoint."""
    if not base_url:
        return []
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        url = base_url.rstrip("/") + "/models"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return [m["id"] for m in response.json().get("data", [])]
        return []
    except Exception:
        return []


def get_custom_openai_config() -> dict:
    """Read custom OpenAI-compatible endpoint settings from the environment."""
    embed_dim = _get_env_int("CUSTOM_OPENAI_EMBED_DIM", 1536)

    return {
        "base_url": (
            os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
            or os.environ.get("CUSTOM_OPENAI_ENDPOINT", "")
        ).rstrip("/"),
        "api_key": os.environ.get("CUSTOM_OPENAI_API_KEY", ""),
        "embed_model": os.environ.get("CUSTOM_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        "embed_dim": embed_dim,
    }


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name, "")
    if raw_value:
        try:
            return int(raw_value)
        except ValueError:
            pass
    return default


def get_openai_compatible_base_url(base_url: str) -> str:
    """Return a base URL ending at the OpenAI-compatible /v1 path."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        return normalized
    return normalized + "/v1"


def get_ollama_config() -> dict:
    """Read Ollama defaults from env/session state."""
    base_url = (
        os.environ.get("OLLAMA_BASE_URL", "")
        or st.session_state.get("ollama_base_url", "http://localhost:11434")
    ).rstrip("/")
    embed_dim = _get_env_int(
        "OLLAMA_EMBED_DIM",
        st.session_state.get("ollama_embed_dim", 768),
    )
    return {
        "base_url": base_url,
        "llm_model": os.environ.get("OLLAMA_LLM_MODEL", "llama3.2:3b"),
        "embed_model": os.environ.get(
            "OLLAMA_EMBED_MODEL",
            st.session_state.get("ollama_embed_model", "nomic-embed-text"),
        ),
        "embed_dim": embed_dim,
    }


def get_preferred_model_index(available_models: list[str]) -> int:
    """Prefer the local Ollama model when it is available."""
    ollama_config = get_ollama_config()
    preferred_model = f"ollama:{ollama_config['llm_model']}"
    if preferred_model in available_models:
        return available_models.index(preferred_model)
    for idx, model in enumerate(available_models):
        if model.startswith("ollama:"):
            return idx
    return 0


def is_ollama_embedding_model(model: str, embed_model: str) -> bool:
    return model == embed_model or model.split(":", 1)[0] == embed_model.split(":", 1)[0]


def get_default_embed_config(ollama_models: list[str], openai_key: str) -> dict:
    """Prefer Ollama embeddings when available, then OpenAI embeddings."""
    ollama_config = get_ollama_config()
    if ollama_models:
        return {
            "available": True,
            "provider": "ollama",
            "embed_model": ollama_config["embed_model"],
            "embed_base_url": get_openai_compatible_base_url(ollama_config["base_url"]),
            "embed_api_key": "ollama",
            "embed_dim": ollama_config["embed_dim"],
        }
    if openai_key:
        return {
            "available": True,
            "provider": "openai",
            "embed_model": "text-embedding-3-small",
            "embed_base_url": "",
            "embed_api_key": openai_key,
            "embed_dim": 1536,
        }
    return {
        "available": False,
        "provider": "",
        "embed_model": "",
        "embed_base_url": "",
        "embed_api_key": "",
        "embed_dim": 0,
    }


@st.cache_data(ttl=300)
def get_available_models() -> dict[str, list[str]]:
    """Get available models for all providers."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    gemini_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    return {
        "anthropic": get_anthropic_models(anthropic_key),
        "openai": get_openai_models(openai_key),
        "groq": get_groq_models(groq_key),
        "gemini": get_gemini_models(gemini_key),
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Pricing data and cost calculation
# ═══════════════════════════════════════════════════════════════════════════════

# Pricing per 1M tokens (USD) - Updated for April 2026
MODEL_PRICING = {
    # Anthropic (Claude)
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00, "provider": "Anthropic"},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00, "provider": "Anthropic"},
    "claude-opus-4-1-20250805": {"input": 15.00, "output": 45.00, "provider": "Anthropic"},
    "claude-opus-1-20250220": {"input": 15.00, "output": 45.00, "provider": "Anthropic"},
    
    # OpenAI (GPT)
    "gpt-4": {"input": 30.00, "output": 60.00, "provider": "OpenAI"},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00, "provider": "OpenAI"},
    "gpt-4o": {"input": 5.00, "output": 15.00, "provider": "OpenAI"},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "provider": "OpenAI"},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "provider": "OpenAI"},
    
    # Groq (very cheap)
    "mixtral-8x7b-32768": {"input": 0.27, "output": 0.81, "provider": "Groq"},
    "llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79, "provider": "Groq"},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.10, "provider": "Groq"},
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79, "provider": "Groq"},
    
    # Google Gemini
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00, "provider": "Google"},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30, "provider": "Google"},
    "gemini-2.0-flash": {"input": 0.075, "output": 0.30, "provider": "Google"},
}

def get_model_pricing(model_name: str) -> dict:
    """Get pricing for a specific model."""
    return MODEL_PRICING.get(model_name, {"input": 0, "output": 0, "provider": "Unknown"})

def estimate_tokens_for_chunk(chunk: str) -> int:
    """Rough estimate of tokens in a chunk (1 token ≈ 4 characters)."""
    return len(chunk) // 4

def estimate_total_tokens(chunks: list[str]) -> tuple[int, int]:
    """Estimate total input and output tokens for processing."""
    # Input tokens: all chunks combined
    input_tokens = sum(estimate_tokens_for_chunk(chunk) for chunk in chunks)
    
    # Output tokens: assume ~200 tokens per chunk for entity extraction
    # (This is a rough estimate; actual will vary)
    output_tokens = len(chunks) * 200
    
    return input_tokens, output_tokens

def calculate_processing_cost(chunks: list[str], model_name: str) -> dict:
    """Calculate estimated cost for processing chunks."""
    pricing = get_model_pricing(model_name)
    input_tokens, output_tokens = estimate_total_tokens(chunks)
    
    input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0)
    total_cost = input_cost + output_cost
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "provider": pricing.get("provider", "Unknown")
    }

def get_all_models_with_pricing() -> pd.DataFrame:
    """Get all available models with pricing info."""
    models_data = []
    for model_name, pricing in MODEL_PRICING.items():
        models_data.append({
            "Model": model_name,
            "Provider": pricing.get("provider", "Unknown"),
            "Input Cost (per 1M tokens)": f"${pricing.get('input', 0):.2f}",
            "Output Cost (per 1M tokens)": f"${pricing.get('output', 0):.2f}",
            "Avg Cost (per 1M tokens)": f"${(pricing.get('input', 0) + pricing.get('output', 0)) / 2:.2f}"
        })
    return pd.DataFrame(models_data).sort_values("Provider")

# ═══════════════════════════════════════════════════════════════════════════════
# Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PDF → Knowledge Graph",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stSidebar"]          { background: #1e293b; }
  .main-title {
    font-size: 2.1rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
  }
  .subtitle { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .stat-box {
    background: #1e293b; border: 1px solid #334155; border-radius: 10px;
    padding: 1rem; text-align: center; margin-bottom: 0.5rem;
  }
  .stat-number { font-size: 1.9rem; font-weight: 700; color: #818cf8; }
  .stat-label  { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
  .chunk-ok {
    background: #0f2a1a; border-left: 3px solid #22c55e; border-radius: 4px;
    padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.8rem; color: #86efac;
  }
  .chunk-err {
    background: #2a0f0f; border-left: 3px solid #ef4444; border-radius: 4px;
    padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.8rem; color: #fca5a5;
  }
  .fact-card {
    background: #1a0f2a; border-left: 3px solid #a855f7; border-radius: 4px;
    padding: 0.5rem 0.8rem; margin-bottom: 0.35rem; font-size: 0.82rem; color: #d8b4fe;
  }
  .search-result {
    background: #0f1f2a; border-left: 3px solid #38bdf8; border-radius: 4px;
    padding: 0.55rem 0.9rem; margin-bottom: 0.4rem; font-size: 0.82rem; color: #7dd3fc;
  }
  h3 { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Session state
# ═══════════════════════════════════════════════════════════════════════════════
_defaults = {
    "graph_built":    False,
    "all_nodes":      {},   # uuid -> EntityNode
    "all_edges":      [],   # list[EntityEdge]
    "episodes_log":   [],
    "pdf_name":       "",
    "db_path":        "",
    "search_results": [],
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


class KognitaKuzuDriver(KuzuDriver):
    """Kuzu driver that also creates Graphiti's full-text indexes."""

    def setup_schema(self):
        super().setup_schema()
        conn = kuzu.Connection(self.db)
        try:
            for query in get_fulltext_indices(GraphProvider.KUZU):
                try:
                    conn.execute(query)
                except RuntimeError as exc:
                    if "already exists" not in str(exc):
                        raise
        finally:
            conn.close()


def _resolve_provider(provider: str, cloud_keys: dict, embed_config: dict) -> dict:
    """Return the full config dict needed to call make_graphiti / ingest_pdf / search_graph.

    cloud_keys = {"anthropic": ..., "openai": ..., "groq": ..., "gemini": ...}
    """
    if provider == "ollama":
        ollama_config = get_ollama_config()
        base = get_openai_compatible_base_url(ollama_config["base_url"])
        return dict(
            api_key="ollama",
            base_url=base,
            embed_model=embed_config["embed_model"],
            embed_base_url=embed_config["embed_base_url"],
            embed_api_key=embed_config["embed_api_key"],
            embed_dim=embed_config["embed_dim"],
        )
    if provider == "custom":
        custom_config = get_custom_openai_config()
        return dict(
            api_key=custom_config["api_key"],
            base_url=custom_config["base_url"],
            embed_model=embed_config["embed_model"],
            embed_base_url=embed_config["embed_base_url"],
            embed_api_key=embed_config["embed_api_key"],
            embed_dim=embed_config["embed_dim"],
        )
    # Cloud providers: LLM key from cloud_keys, embeddings from preferred embed config.
    return dict(
        api_key=cloud_keys.get(provider, ""),
        base_url="",
        embed_model=embed_config["embed_model"],
        embed_base_url=embed_config["embed_base_url"],
        embed_api_key=embed_config["embed_api_key"],
        embed_dim=embed_config["embed_dim"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Health Check")
    ollama_config = get_ollama_config()
    ollama_models = get_ollama_models(ollama_config["base_url"])
    ollama_embed_model = ollama_config["embed_model"]
    ollama_llm_models = [
        model for model in ollama_models
        if not is_ollama_embedding_model(model, ollama_embed_model)
    ]

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.write(f"Anthropic API Key: {'✅' if anthropic_key else '❌'}")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    st.write(f"OpenAI API Key: {'✅' if openai_key else '❌'}")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    st.write(f"Groq API Key: {'✅' if groq_key else '❌'}")
    gemini_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    st.write(f"Gemini API Key: {'✅' if gemini_key else '❌'}")
    if ollama_models:
        st.write(f"Ollama: ✅ connected · {len(ollama_llm_models)} LLMs")
    else:
        st.write("Ollama: ❌ not reachable")
    custom_config = get_custom_openai_config()
    if custom_config["base_url"] or custom_config["api_key"]:
        custom_endpoint_status = f"✅ {custom_config['base_url']}" if custom_config["base_url"] else "❌"
        st.write(f"Custom OpenAI Endpoint: {custom_endpoint_status}")
        st.write(f"Custom OpenAI API Key: {'✅' if custom_config['api_key'] else '❌'}")
    embed_config = get_default_embed_config(ollama_models, openai_key)
    if embed_config["available"]:
        st.write(f"Embeddings: ✅ {embed_config['provider']} · {embed_config['embed_model']}")
    else:
        st.error("Embeddings: ❌ Start Ollama or set OPENAI_API_KEY")

    custom_models = (
        get_custom_models(custom_config["base_url"], custom_config["api_key"])
        if custom_config["base_url"]
        else []
    )

    # ── Available models based on all configured providers ────────────────────
    st.divider()
    available_models_dict = get_available_models()
    available_models = [
        f"{prov}:{m}"
        for prov, ms in available_models_dict.items()
        for m in ms
    ]
    available_models += [f"ollama:{m}" for m in ollama_llm_models]
    available_models += [f"custom:{m}" for m in custom_models]

    if available_models:
        st.success(f"✅ {len(available_models)} models available")
    else:
        st.error("❌ No providers configured")

    st.divider()
    st.markdown("## 🤖 Graph Processing Model")

    # Model selection for graph processing
    if available_models:
        processing_model = st.selectbox(
            "Model for Knowledge Graph Building",
            available_models,
            index=get_preferred_model_index(available_models),
            help="Choose which LLM to use for extracting entities and relationships from PDF chunks"
        )

        # Store the selected processing model in session state
        st.session_state.processing_model = processing_model
        
        # Extract actual model ID from "provider:model-id"
        actual_model_name = processing_model.split(":", 1)[1] if ":" in processing_model else processing_model
        
        # Display pricing for selected model
        pricing_info = get_model_pricing(actual_model_name)
        if pricing_info.get("input", 0) > 0:
            st.metric(
                "💰 Cost per 1M tokens",
                f"Input: ${pricing_info['input']:.2f} | Output: ${pricing_info['output']:.2f}"
            )
        
        # Add button to show pricing modal
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("📊", help="View pricing for all available models"):
                st.session_state.show_pricing_modal = True
    else:
        st.error("No models available for processing")
        processing_model = None
        st.session_state.processing_model = None

    st.divider()

    # Load saved graphs
    st.markdown("## 💾 Saved Graphs")
    saved_graphs = get_saved_graphs()

    if saved_graphs:
        graph_options = ["Select a saved graph..."] + [f"{g['pdf_name']} ({g['processed_at'][:10]}) - {g['node_count']} nodes" for g in saved_graphs]
        selected_graph = st.selectbox(
            "Load previously processed graph",
            graph_options,
            help="Load a previously processed knowledge graph to avoid re-processing the same PDF"
        )

        if selected_graph != "Select a saved graph...":
            graph_index = graph_options.index(selected_graph) - 1
            selected_metadata = saved_graphs[graph_index]

            if st.button("🔄 Load Selected Graph", type="primary"):
                with st.spinner("Loading saved graph..."):
                    try:
                        nodes, edges, episodes, pdf_bytes, pdf_name = load_graph_data(selected_metadata["graph_dir"])

                        # Update session state
                        st.session_state.nodes = nodes
                        st.session_state.edges = edges
                        st.session_state.episodes_log = episodes
                        st.session_state.pdf_bytes = pdf_bytes
                        st.session_state.pdf_name = pdf_name
                        st.session_state.graph_built = True
                        st.session_state.processing_model = selected_metadata.get("processing_model", "Unknown")

                        st.success(f"✅ Loaded graph for '{pdf_name}' with {len(nodes)} nodes and {len(edges)} edges")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load graph: {str(e)}")
    else:
        st.info("No saved graphs found. Process a PDF first to create saved graphs.")

    st.divider()
    st.markdown("## 📄 Chunking")
    chunk_size    = st.slider("Words per chunk",  80, 500, 220, 20)
    chunk_overlap = st.slider("Overlap words",     0,  80,  25,  5)
    if chunk_overlap >= chunk_size:
        st.error("⚠️ Overlap must be less than chunk size.")

    st.divider()
    st.markdown("## 🎨 Visualisation")
    node_color  = st.color_picker("Node colour",  "#818cf8")
    edge_color  = st.color_picker("Edge colour",  "#a855f7")
    physics_on  = st.toggle("Physics simulation", True)
    show_labels = st.toggle("Show edge labels",   True)

# ═══════════════════════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🧠 PDF → Knowledge Graph</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Powered by Graphiti · KuzuDB · Claude Sonnet · '
    'text-embedding-3-small</div>',
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Core helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    step = max(1, size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + size])
        if len(chunk.strip()) > 60:
            chunks.append(chunk)
        i += step
    return chunks


def extract_model_name(display_name: str) -> str:
    """Extract the model ID from a display name.

    Handles the canonical format: "provider:model-id" → "model-id"
    """
    if ":" in display_name:
        return display_name.split(":", 1)[1]
    return display_name

def make_graphiti(
    provider: str,
    api_key: str,
    db_path: str,
    model: str,
    base_url: str = "",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "",
    embed_api_key: str = "",
    embed_dim: int = 1536,
) -> Graphiti:
    """Create a Graphiti instance with the specified LLM provider and model.

    For cloud providers (anthropic, openai, groq, gemini) leave base_url empty.
    For ollama/custom pass the full base URL (e.g. http://localhost:11434/v1).
    embed_* params configure the embedding model independently of the LLM provider.
    """
    actual_model = extract_model_name(model)

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=embed_api_key or os.environ.get("OPENAI_API_KEY", ""),
            embedding_model=embed_model,
            embedding_dim=embed_dim,
            base_url=embed_base_url or None,
        )
    )

    if provider == "anthropic":
        llm_client = AnthropicClient(config=LLMConfig(api_key=api_key, model=actual_model))
        cross_encoder = None
    elif provider == "openai":
        llm_client = OpenAIClient(config=LLMConfig(api_key=api_key, model=actual_model))
        cross_encoder = None
    elif provider == "groq":
        llm_config = LLMConfig(
            api_key=api_key,
            model=actual_model,
            small_model=actual_model,
            base_url="https://api.groq.com/openai/v1",
        )
        llm_client = OpenAIGenericClient(config=llm_config)
        cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=llm_config)
    elif provider == "gemini":
        llm_client = GeminiClient(config=LLMConfig(api_key=api_key, model=actual_model))
        cross_encoder = None
    elif provider in ("ollama", "custom"):
        llm_config = LLMConfig(
            api_key=api_key or "ollama",
            model=actual_model,
            small_model=actual_model,
            base_url=base_url,
        )
        llm_client = OpenAIGenericClient(
            config=llm_config
        )
        cross_encoder = OpenAIRerankerClient(client=llm_client.client, config=llm_config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    graphiti_kwargs = {}
    if cross_encoder is not None:
        graphiti_kwargs["cross_encoder"] = cross_encoder

    return Graphiti(
        graph_driver=KognitaKuzuDriver(db=db_path),
        llm_client=llm_client,
        embedder=embedder,
        **graphiti_kwargs,
    )


def _extract_api_error(exc: Exception) -> str:
    """Return a human-readable error from a provider API exception.

    Handles OpenAI / Groq (openai.APIStatusError), Anthropic (anthropic.APIStatusError),
    and Google Gemini (google.api_core.exceptions.GoogleAPICallError), as well as any
    generic exception that carries a status_code attribute.
    """
    status_code = getattr(exc, "status_code", None)  # OpenAI, Anthropic
    if status_code is None:
        status_code = getattr(exc, "code", None)      # Gemini / gRPC style

    # Prefer the structured body/message over str(exc) which may include noisy traceback
    body = getattr(exc, "body", None)
    if body:
        detail = str(body)
    else:
        detail = getattr(exc, "message", None) or str(exc)

    if status_code:
        return f"HTTP {status_code}: {detail}"
    return f"{type(exc).__name__}: {detail}"


def _node_log_line(node: EntityNode) -> str:
    name = getattr(node, "name", None) or getattr(node, "uuid", "")[:10] or "(unnamed)"
    labels = getattr(node, "labels", None) or []
    labels_text = f" [{', '.join(str(label) for label in labels)}]" if labels else ""
    summary = (getattr(node, "summary", None) or "").replace("\n", " ").strip()
    summary_text = f" - {summary[:140]}" if summary else ""
    return f"Node added: {name}{labels_text}{summary_text}"


def _edge_log_line(edge: EntityEdge, nodes: dict[str, EntityNode]) -> str:
    source_node = nodes.get(getattr(edge, "source_node_uuid", ""))
    target_node = nodes.get(getattr(edge, "target_node_uuid", ""))
    source = getattr(source_node, "name", None) or getattr(edge, "source_node_uuid", "")[:10]
    target = getattr(target_node, "name", None) or getattr(edge, "target_node_uuid", "")[:10]
    fact = (getattr(edge, "fact", None) or getattr(edge, "name", None) or "").replace("\n", " ").strip()
    fact_text = f" - {fact[:180]}" if fact else ""
    return f"Edge added: {source} -> {target}{fact_text}"


def get_next_chunk_index(episodes_log: list[dict]) -> int:
    """Return the zero-based chunk index to resume from."""
    processed_chunks = [
        int(ep.get("chunk", 0))
        for ep in episodes_log
        if str(ep.get("chunk", "")).isdigit()
    ]
    return max(processed_chunks, default=0)


async def ingest_pdf(
    chunks: list[str],
    pdf_name: str,
    provider: str,
    api_key: str,
    model: str,
    db_path: str,
    progress_cb,
    status_cb,
    log_cb=None,
    base_url: str = "",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "",
    embed_api_key: str = "",
    embed_dim: int = 1536,
    start_index: int = 0,
    initial_nodes: dict[str, EntityNode] | None = None,
    initial_edges: list[EntityEdge] | None = None,
    initial_episodes_log: list[dict] | None = None,
) -> tuple[dict[str, EntityNode], list[EntityEdge], list[dict], str | None]:
    """Process PDF chunks into a knowledge graph.

    Returns (nodes, edges, episodes_log, fatal_error).
    fatal_error is None on success; a non-empty string on any provider API error.
    Processing stops immediately on the first error.
    """
    graphiti = make_graphiti(provider, api_key, db_path, model,
                             base_url, embed_model, embed_base_url, embed_api_key, embed_dim)
    await graphiti.build_indices_and_constraints()

    all_nodes: dict[str, EntityNode] = dict(initial_nodes or {})
    all_edges: list[EntityEdge] = list(initial_edges or [])
    episodes_log: list[dict] = list(initial_episodes_log or [])
    n = len(chunks)

    for idx, chunk in enumerate(chunks[start_index:], start=start_index):
        chunk_label = f"Chunk {idx + 1}/{n}"
        status_cb(f"⚙️  Processing {chunk_label} …")
        if log_cb:
            log_cb(f"{chunk_label}: sending text to Graphiti")
        try:
            result = await graphiti.add_episode(
                name=f"{pdf_name}__chunk_{idx + 1:04d}",
                episode_body=chunk,
                source=EpisodeType.text,
                source_description=f"PDF: {pdf_name}",
                reference_time=datetime.now(),
            )
            if log_cb:
                log_cb(
                    f"{chunk_label}: Graphiti returned "
                    f"{len(result.nodes)} nodes and {len(result.edges)} edges"
                )
            for node in result.nodes:
                all_nodes[node.uuid] = node
                if log_cb:
                    log_cb(_node_log_line(node))
            all_edges.extend(result.edges)
            for edge in result.edges:
                if log_cb:
                    log_cb(_edge_log_line(edge, all_nodes))
            episodes_log.append({
                "chunk":   idx + 1,
                "preview": chunk[:130].replace("\n", " ") + "…",
                "nodes":   len(result.nodes),
                "edges":   len(result.edges),
                "node_log": [_node_log_line(node) for node in result.nodes],
                "edge_log": [_edge_log_line(edge, all_nodes) for edge in result.edges],
            })
        except Exception as exc:
            error_msg = _extract_api_error(exc)
            episodes_log.append({
                "chunk":   idx + 1,
                "preview": chunk[:130].replace("\n", " ") + "…",
                "error":   error_msg,
                "nodes":   0,
                "edges":   0,
            })
            if log_cb:
                log_cb(f"{chunk_label}: stopped with error - {error_msg}")
            status_cb(f"❌ Stopped at chunk {idx + 1} of {n}: {error_msg}")
            await graphiti.close()
            return all_nodes, all_edges, episodes_log, error_msg
        progress_cb((idx + 1) / n)

    await graphiti.close()
    return all_nodes, all_edges, episodes_log, None


async def search_graph(
    query: str, provider: str, api_key: str, model: str, db_path: str,
    base_url: str = "", embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "", embed_api_key: str = "", embed_dim: int = 1536,
) -> list:
    graphiti = make_graphiti(provider, api_key, db_path, model,
                             base_url, embed_model, embed_base_url, embed_api_key, embed_dim)
    try:
        results = await graphiti.search(query)
    except Exception as exc:
        await graphiti.close()
        raise RuntimeError(_extract_api_error(exc)) from exc
    await graphiti.close()
    return results


def run_async(coro):
    return asyncio.run(coro)


def build_pyvis_html(
    nodes: dict[str, EntityNode],
    edges: list[EntityEdge],
    n_color: str,
    e_color: str,
    physics: bool,
    edge_labels: bool,
) -> str:
    net = Network(
        height="650px", width="100%",
        bgcolor="#0f172a", font_color="#e2e8f0",
        directed=True,
    )
    if physics:
        net.barnes_hut(
            gravity=-6000, central_gravity=0.25,
            spring_length=140, spring_strength=0.04, damping=0.09,
        )
    else:
        net.toggle_physics(False)

    # Node degree for sizing
    degree: Counter = Counter()
    for e in edges:
        degree[e.source_node_uuid] += 1
        degree[e.target_node_uuid] += 1

    for uuid, node in nodes.items():
        label = (node.name or uuid[:12])[:40]
        size  = 14 + min(degree.get(uuid, 0) * 3, 30)
        tip   = f"<b>{node.name}</b>"
        if node.summary:
            tip += f"<br><br>{node.summary[:300]}"
        if node.labels:
            tip += f"<br><br><i>Labels: {', '.join(node.labels)}</i>"
        net.add_node(
            uuid, label=label, title=tip,
            color={
                "background": n_color,
                "border": "#c4b5fd",
                "highlight": {"background": "#c084fc", "border": "#e879f9"},
            },
            size=size,
            font={"size": 11, "color": "#f1f5f9"},
            borderWidth=2,
        )

    node_set = set(nodes.keys())
    seen: set = set()
    for edge in edges:
        src, tgt = edge.source_node_uuid, edge.target_node_uuid
        if src not in node_set or tgt not in node_set:
            continue
        key = (src, tgt, (edge.fact or "")[:40])
        if key in seen:
            continue
        seen.add(key)
        tip = edge.fact or edge.name or ""
        lbl = (tip[:28] + "…") if edge_labels and len(tip) > 28 else (tip if edge_labels else "")
        net.add_edge(
            src, tgt,
            title=tip, label=lbl,
            color={"color": e_color, "opacity": 0.75, "highlight": "#e879f9"},
            arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
            smooth={"type": "curvedCW", "roundness": 0.15},
            font={"size": 9, "color": "#c4b5fd", "strokeWidth": 0},
            width=1.5,
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    tmp.close()
    try:
        net.save_graph(tmp.name)
        return Path(tmp.name).read_text(encoding="utf-8")
    finally:
        os.unlink(tmp.name)


async def generate_llm_response(
    question: str,
    search_results: list,
    provider: str,
    api_key: str,
    nodes: dict,
    edges: list,
    selected_model: str,
    base_url: str = "",
) -> str:
    """Generate a response using LLM based on search results and graph context."""

    # Create context from search results
    context = f"Question: {question}\n\n"
    context += f"Knowledge Graph Summary: {len(nodes)} entities, {len(edges)} relationships\n\n"

    if search_results:
        context += "Relevant information from the knowledge graph:\n"
        for i, result in enumerate(search_results[:5]):
            context += f"{i+1}. {str(result)}\n"
        context += "\n"

    # Create a prompt for the LLM
    prompt = f"""You are a helpful assistant that answers questions about a knowledge graph.

{context}

Please provide a comprehensive and accurate answer to the user's question based on the information available in the knowledge graph. If the information is insufficient, acknowledge this and suggest what might be needed.

Answer:"""

    try:
        if provider == "anthropic":
            actual_model = extract_model_name(selected_model) or "claude-3-5-sonnet-20241022"

            llm_client = AnthropicClient(
                config=LLMConfig(
                    api_key=api_key,
                    model=actual_model,
                )
            )
            response = await llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.content[0].text if response.content else "No response generated."

        elif provider == "openai":
            actual_model = extract_model_name(selected_model) or "gpt-4"

            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content if response.choices else "No response generated."

        elif provider == "groq":
            actual_model = extract_model_name(selected_model) or "llama3-70b-8192"

            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = await client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content if response.choices else "No response generated."

        elif provider == "gemini":
            if genai is None:
                return "Gemini library not available. Please install google-generativeai."

            actual_model = extract_model_name(selected_model) or "gemini-1.5-pro"

            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(actual_model)
            response = await gemini_model.generate_content_async(prompt)
            return response.text if response.text else "No response generated."

        elif provider in ("ollama", "custom"):
            actual_model = extract_model_name(selected_model)
            client = openai.AsyncOpenAI(
                api_key=api_key or "ollama",
                base_url=base_url,
            )
            response = await client.chat.completions.create(
                model=actual_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )
            return response.choices[0].message.content if response.choices else "No response generated."

        else:
            return f"Unsupported provider: {provider}"

    except Exception as exc:
        return f"Error generating response: {_extract_api_error(exc)}"


def execute_kuzu_query(query: str, db_path: str) -> list:
    """Execute a direct Kuzu query and return results.

    NOTE: Direct Kuzu query execution is not yet implemented.
    Use the semantic search in the Search Graph section instead.
    """
    raise NotImplementedError(
        f"Direct Kuzu query execution is not yet implemented (query: {query!r}). "
        "Use the 'Search Graph' section for semantic queries against the knowledge graph."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Pricing Modal Dialog
# ═══════════════════════════════════════════════════════════════════════════════
if "show_pricing_modal" not in st.session_state:
    st.session_state.show_pricing_modal = False

if st.session_state.show_pricing_modal:
    with st.expander("📊 **Model Pricing Comparison**", expanded=True):
        st.markdown("### All Available Models and Their Costs")
        pricing_df = get_all_models_with_pricing()
        st.dataframe(
            pricing_df,
            use_container_width=True,
            hide_index=True,
        )
        st.info("💡 **Tip:** Groq models are the cheapest option for most tasks. Click 'See All Pricing' again to close this view.")
        if st.button("Close Pricing View"):
            st.session_state.show_pricing_modal = False
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# Two-column layout
# ═══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2], gap="large")

# ─── LEFT ────────────────────────────────────────────────────────────────────
with left:
    st.markdown("### 1 · Upload PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        pdf_bytes = uploaded.read()
        st.success(f"✅ **{uploaded.name}** · {len(pdf_bytes) // 1024} KB")

        # Check if PDF has already been processed
        existing_graph_dir = is_pdf_already_processed(pdf_bytes)
        if existing_graph_dir:
            st.info("📁 **This PDF has already been processed!**")
            if st.button("🔄 Load Existing Graph", type="secondary"):
                with st.spinner("Loading saved graph..."):
                    try:
                        nodes, edges, episodes, _, _ = load_graph_data(existing_graph_dir)

                        # Update session state
                        st.session_state.all_nodes = nodes
                        st.session_state.all_edges = edges
                        st.session_state.episodes_log = episodes
                        st.session_state.graph_built = True

                        # Load metadata to get processing model
                        with open(os.path.join(existing_graph_dir, "metadata.json"), 'r') as f:
                            metadata = json.load(f)
                        st.session_state.processing_model = metadata.get("processing_model", "Unknown")

                        st.success("✅ Loaded existing graph!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load existing graph: {str(e)}")
        else:
            st.info("🆕 **New PDF detected** - ready for processing")

        raw_text = extract_text(pdf_bytes)
        chunks   = chunk_text(raw_text, chunk_size, chunk_overlap)

        with st.expander(f"Preview extracted text · {len(raw_text):,} chars"):
            st.text_area(
                "Extracted text preview", value=raw_text[:3000] + "\n…",
                height=200, label_visibility="collapsed", disabled=True,
            )

        st.caption(f"→ **{len(chunks)} chunks** · ~{chunk_size} words · "
                   f"{chunk_overlap} word overlap")

        st.markdown("### 2 · Build")
        keys_ok = bool(available_models and embed_config["available"])
        if not available_models:
            st.warning("⚠️ Configure at least one provider in the sidebar first.")
        elif not embed_config["available"]:
            st.error("❌ Start Ollama or set OPENAI_API_KEY before building the graph.")

        if st.button(
            "🚀 Build Knowledge Graph",
            disabled=not keys_ok,
            use_container_width=True,
            type="primary",
        ):
            db_dir  = tempfile.mkdtemp()
            db_path = os.path.join(db_dir, "kg")
            st.session_state.db_path  = db_path
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_bytes = pdf_bytes  # Store PDF bytes for saving

            # Reset
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v
            st.session_state.db_path  = db_path
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_bytes = pdf_bytes

            prog = st.progress(0.0)
            stat = st.empty()
            live_log = st.empty()
            processing_log: list[str] = []

            def append_processing_log(message: str):
                timestamp = datetime.now().strftime("%H:%M:%S")
                processing_log.append(f"[{timestamp}] {message}")
                visible_log = "\n".join(processing_log[-160:])
                live_log.text_area(
                    "Live processing log",
                    value=visible_log,
                    height=320,
                    disabled=True,
                    help="Updates after each Graphiti step, node, and edge.",
                )

            # Get processing model details
            processing_model = st.session_state.get("processing_model")
            if not processing_model:
                st.error("No processing model selected")
                st.stop()

            processing_provider = processing_model.split(":", 1)[0]
            cloud_keys = {"anthropic": anthropic_key, "openai": openai_key,
                          "groq": groq_key, "gemini": gemini_key}
            pconf = _resolve_provider(processing_provider, cloud_keys, embed_config)

            with st.spinner("Graphiti is working …"):
                nodes, edges, ep_log, fatal_error = run_async(
                    ingest_pdf(
                        chunks, uploaded.name,
                        processing_provider, pconf["api_key"], processing_model, db_path,
                        lambda v: prog.progress(v),
                        lambda m: stat.info(m),
                        append_processing_log,
                        base_url=pconf["base_url"],
                        embed_model=pconf["embed_model"],
                        embed_base_url=pconf["embed_base_url"],
                        embed_api_key=pconf["embed_api_key"],
                        embed_dim=pconf["embed_dim"],
                    )
                )

            st.session_state.all_nodes    = nodes
            st.session_state.all_edges    = edges
            st.session_state.episodes_log = ep_log
            st.session_state.graph_built  = True

            if fatal_error:
                stat.error(f"❌ Processing stopped — provider returned an error:")
                st.error(fatal_error)
            else:
                stat.success("✅ Graph built successfully!")
                try:
                    save_graph_data(
                        uploaded.name, pdf_bytes, nodes, edges, ep_log, processing_model
                    )
                    st.info("💾 Graph automatically saved for future use")
                except Exception as e:
                    st.warning(f"⚠️ Failed to save graph: {str(e)}")

        if st.session_state.graph_built and st.session_state.db_path:
            next_chunk_index = get_next_chunk_index(st.session_state.episodes_log)
            can_continue = next_chunk_index < len(chunks)
            if can_continue:
                st.info(
                    f"Processing can continue from chunk {next_chunk_index + 1} "
                    f"of {len(chunks)}. The failed chunk stays in the log."
                )
            if can_continue and st.button(
                f"▶️ Continue from chunk {next_chunk_index + 1}",
                disabled=not keys_ok,
                use_container_width=True,
            ):
                prog = st.progress(next_chunk_index / max(1, len(chunks)))
                stat = st.empty()
                live_log = st.empty()
                processing_log: list[str] = []

                def append_processing_log(message: str):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    processing_log.append(f"[{timestamp}] {message}")
                    live_log.text_area(
                        "Live processing log",
                        value="\n".join(processing_log[-160:]),
                        height=320,
                        disabled=True,
                        help="Updates after each Graphiti step, node, and edge.",
                    )

                processing_model = st.session_state.get("processing_model")
                processing_provider = processing_model.split(":", 1)[0]
                cloud_keys = {"anthropic": anthropic_key, "openai": openai_key,
                              "groq": groq_key, "gemini": gemini_key}
                pconf = _resolve_provider(processing_provider, cloud_keys, embed_config)

                with st.spinner("Graphiti is continuing …"):
                    nodes, edges, ep_log, fatal_error = run_async(
                        ingest_pdf(
                            chunks, uploaded.name,
                            processing_provider, pconf["api_key"], processing_model,
                            st.session_state.db_path,
                            lambda v: prog.progress(v),
                            lambda m: stat.info(m),
                            append_processing_log,
                            base_url=pconf["base_url"],
                            embed_model=pconf["embed_model"],
                            embed_base_url=pconf["embed_base_url"],
                            embed_api_key=pconf["embed_api_key"],
                            embed_dim=pconf["embed_dim"],
                            start_index=next_chunk_index,
                            initial_nodes=st.session_state.all_nodes,
                            initial_edges=st.session_state.all_edges,
                            initial_episodes_log=st.session_state.episodes_log,
                        )
                    )

                st.session_state.all_nodes = nodes
                st.session_state.all_edges = edges
                st.session_state.episodes_log = ep_log

                if fatal_error:
                    stat.error("❌ Processing stopped again — provider returned an error:")
                    st.error(fatal_error)
                else:
                    stat.success("✅ Remaining chunks processed successfully!")
                    try:
                        save_graph_data(
                            uploaded.name, pdf_bytes, nodes, edges, ep_log, processing_model
                        )
                        st.info("💾 Graph automatically saved for future use")
                    except Exception as e:
                        st.warning(f"⚠️ Failed to save graph: {str(e)}")

    # Stats & search (shown after build)
    if st.session_state.graph_built:
        nodes  = st.session_state.all_nodes
        edges  = st.session_state.all_edges
        ep_log = st.session_state.episodes_log
        ok_ep  = sum(1 for e in ep_log if "error" not in e)
        err_ep = len(ep_log) - ok_ep

        st.markdown("### 3 · Stats")
        c1, c2 = st.columns(2)
        stats = [
            (len(nodes),    "Entities",       c1),
            (len(edges),    "Relationships",  c2),
            (ok_ep,         "Episodes OK",    c1),
            (err_ep,        "Episodes Failed",c2),
        ]
        for val, label, col in stats:
            with col:
                st.markdown(
                    f'<div class="stat-box">'
                    f'<div class="stat-number">{val}</div>'
                    f'<div class="stat-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 4 · Search Graph")
        query = st.text_input(
            "Natural language query",
            placeholder="e.g. Who are the key people?",
        )
        if not embed_config["available"]:
            st.error("❌ Search needs embeddings. Start Ollama or set OPENAI_API_KEY.")
        if st.button(
            "🔍 Search",
            use_container_width=True,
            disabled=not embed_config["available"],
        ) and query.strip():
            with st.spinner("Searching …"):
                try:
                    processing_model = st.session_state.get("processing_model", "")
                    search_provider = processing_model.split(":", 1)[0] if ":" in processing_model else "anthropic"
                    cloud_keys = {"anthropic": anthropic_key, "openai": openai_key,
                                  "groq": groq_key, "gemini": gemini_key}
                    sconf = _resolve_provider(search_provider, cloud_keys, embed_config)

                    results = run_async(
                        search_graph(
                            query, search_provider, sconf["api_key"], processing_model,
                            st.session_state.db_path,
                            base_url=sconf["base_url"],
                            embed_model=sconf["embed_model"],
                            embed_base_url=sconf["embed_base_url"],
                            embed_api_key=sconf["embed_api_key"],
                            embed_dim=sconf["embed_dim"],
                        )
                    )
                    st.session_state.search_results = results
                except Exception as exc:
                    st.error(f"Search error: {exc}")

        for r in st.session_state.search_results[:10]:
            fact = getattr(r, "fact", None) or str(r)
            st.markdown(
                f'<div class="search-result">🔗 {fact}</div>',
                unsafe_allow_html=True,
            )

# ─── RIGHT ───────────────────────────────────────────────────────────────────
with right:
    if st.session_state.graph_built:
        nodes  = st.session_state.all_nodes
        edges  = st.session_state.all_edges
        ep_log = st.session_state.episodes_log

        # Option menu for navigation
        selected = option_menu(
            menu_title=None,
            options=["🕸️ Graph", "📋 Episode Log", "📜 All Facts", "📥 Export", "🤖 LLM Playground"],
            icons=["graph-up", "list-check", "file-text", "download", "robot"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#1e293b"},
                "icon": {"color": "#818cf8", "font-size": "16px"},
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#334155",
                },
                "nav-link-selected": {"background-color": "#818cf8"},
            }
        )

        # Graph
        if selected == "🕸️ Graph":
            if nodes:
                st.caption(
                    f"**{len(nodes)} entities** · **{len(edges)} relationships** — "
                    "Hover for details · Drag · Scroll to zoom"
                )
                html = build_pyvis_html(
                    nodes, edges, node_color, edge_color, physics_on, show_labels
                )
                components.html(html, height=680, scrolling=False)
            else:
                st.info(
                    "No entities extracted. Try a content-rich PDF or reduce chunk size."
                )

        # Episode log
        elif selected == "📋 Episode Log":
            st.caption(f"{len(ep_log)} chunks processed")
            for ep in ep_log:
                if "error" in ep:
                    st.markdown(
                        f'<div class="chunk-err">❌ <b>Chunk {ep["chunk"]}</b> — '
                        f'{ep["error"]}<br>'
                        f'<span style="opacity:.6">{ep["preview"]}</span></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chunk-ok">✅ <b>Chunk {ep["chunk"]}</b> — '
                        f'{ep["nodes"]} entities · {ep["edges"]} edges<br>'
                        f'<span style="opacity:.7">{ep["preview"]}</span></div>',
                        unsafe_allow_html=True,
                    )
                    node_log = ep.get("node_log", [])
                    edge_log = ep.get("edge_log", [])
                    if node_log or edge_log:
                        with st.expander(f"Chunk {ep['chunk']} node and edge log"):
                            for line in node_log:
                                st.markdown(f"- `{line}`")
                            for line in edge_log:
                                st.markdown(f"- `{line}`")

        # All facts
        elif selected == "📜 All Facts":
            st.caption(f"{len(edges)} total facts extracted")
            for edge in edges:
                src_node = nodes.get(edge.source_node_uuid)
                tgt_node = nodes.get(edge.target_node_uuid)
                src_lbl  = src_node.name if src_node else edge.source_node_uuid[:10]
                tgt_lbl  = tgt_node.name if tgt_node else edge.target_node_uuid[:10]
                fact     = edge.fact or edge.name or "(unnamed)"
                st.markdown(
                    f'<div class="fact-card">'
                    f'<b>{src_lbl}</b> → <b>{tgt_lbl}</b><br>{fact}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Export
        elif selected == "📥 Export":
            export_data = {
                "pdf":      st.session_state.pdf_name,
                "built_at": datetime.now().isoformat(),
                "nodes": [
                    {
                        "uuid":    n.uuid,
                        "name":    n.name,
                        "summary": n.summary,
                        "labels":  n.labels,
                    }
                    for n in nodes.values()
                ],
                "edges": [
                    {
                        "uuid":   e.uuid,
                        "source": e.source_node_uuid,
                        "target": e.target_node_uuid,
                        "fact":   e.fact,
                        "name":   e.name,
                    }
                    for e in edges
                ],
            }
            st.download_button(
                label="⬇️  Download knowledge_graph.json",
                data=json.dumps(export_data, indent=2, default=str),
                file_name="knowledge_graph.json",
                mime="application/json",
                use_container_width=True,
            )
            st.markdown("#### Entity list")
            for n in sorted(nodes.values(), key=lambda x: x.name or ""):
                summary_snippet = f" — {n.summary[:100]}…" if n.summary else ""
                st.markdown(f"- **{n.name}**{summary_snippet}")

        # LLM Playground
        elif selected == "🤖 LLM Playground":
            st.markdown("### 🤖 LLM Playground")
            st.markdown("Interact with your knowledge graph using LLMs and explore Kuzu database functionalities.")

            # Reuse the same available_models list built in the sidebar
            # (already includes cloud + ollama + custom)

            if not available_models:
                st.error("❌ No API keys configured or failed to fetch models.")
                st.stop()

            # Model Selection
            selected_model = st.selectbox(
                "Choose Model",
                available_models,
                help="Select from available models based on your configured API keys"
            )

            provider = selected_model.split(":", 1)[0]
            cloud_keys = {"anthropic": anthropic_key, "openai": openai_key,
                          "groq": groq_key, "gemini": gemini_key}
            pconf = _resolve_provider(provider, cloud_keys, embed_config)
            api_key = pconf["api_key"]

            # Chat Interface
            st.markdown("#### 💬 Chat with Knowledge Graph")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Display chat history

            # Display chat history
            chat_container = st.container(height=300)
            with chat_container:
                for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
                    if msg["role"] == "user":
                        st.markdown(f"**You:** {msg['content']}")
                    else:
                        st.markdown(f"**Assistant:** {msg['content']}")
                        if "search_results" in msg:
                            with st.expander("📊 Search Results"):
                                for result in msg["search_results"][:5]:
                                    st.write(f"• {result}")

            # Chat input
            chat_input = st.text_input(
                "Ask a question about your knowledge graph:",
                placeholder="e.g., 'What are the main topics discussed?' or 'Who are the key entities?'",
                key="chat_input"
            )

            if st.button("💬 Send", use_container_width=True) and chat_input.strip():
                if provider not in ("ollama", "custom") and not api_key:
                    st.error("Please set the API key for the selected LLM provider.")
                elif not embed_config["available"]:
                    st.error("Please start Ollama or set OPENAI_API_KEY before searching the graph.")
                else:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": chat_input})

                    with st.spinner("Thinking..."):
                        try:
                            processing_model = st.session_state.get("processing_model", "")
                            search_provider = processing_model.split(":", 1)[0] if ":" in processing_model else provider
                            sconf = _resolve_provider(search_provider, cloud_keys, embed_config)
                            search_results = run_async(
                                search_graph(
                                    chat_input, search_provider, sconf["api_key"],
                                    processing_model, st.session_state.db_path,
                                    base_url=sconf["base_url"],
                                    embed_model=sconf["embed_model"],
                                    embed_base_url=sconf["embed_base_url"],
                                    embed_api_key=sconf["embed_api_key"],
                                    embed_dim=sconf["embed_dim"],
                                )
                            )

                            # Generate response using LLM
                            response = run_async(
                                generate_llm_response(
                                    chat_input, search_results, provider, api_key,
                                    nodes, edges, selected_model,
                                    base_url=pconf["base_url"],
                                )
                            )

                            # Add assistant response to history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "search_results": search_results
                            })

                            st.rerun()

                        except Exception as exc:
                            st.error(f"Error: {exc}")

            # Kuzu Database Explorer
            st.markdown("#### 🗄️ Kuzu Database Explorer")
            st.markdown("Execute direct queries on the Kuzu graph database.")

            # Predefined queries
            predefined_queries = {
                "Count all nodes": "MATCH (n) RETURN count(n) as node_count",
                "Count all edges": "MATCH ()-[r]->() RETURN count(r) as edge_count",
                "List all node labels": "MATCH (n) RETURN DISTINCT labels(n) as labels",
                "Find nodes with most connections": "MATCH (n)-[r]-() RETURN n.name, count(r) as connections ORDER BY connections DESC LIMIT 10",
                "Show recent episodes": "MATCH (e:Episode) RETURN e.name, e.created_at ORDER BY e.created_at DESC LIMIT 5"
            }

            query_option = st.selectbox(
                "Choose a query or write your own:",
                ["Custom Query"] + list(predefined_queries.keys())
            )

            if query_option == "Custom Query":
                kuzu_query = st.text_area(
                    "Kuzu Cypher Query:",
                    height=100,
                    placeholder="MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10"
                )
            else:
                kuzu_query = st.text_area(
                    "Kuzu Cypher Query:",
                    value=predefined_queries[query_option],
                    height=100
                )

            if st.button("🔍 Execute Query", use_container_width=True) and kuzu_query.strip():
                try:
                    results = execute_kuzu_query(kuzu_query, st.session_state.db_path)
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)
                except NotImplementedError as exc:
                    st.warning(f"⚠️ {exc}")
                except Exception as exc:
                    st.error(f"Query error: {exc}")

            # Example Queries Section
            with st.expander("📚 Example Queries & Tips"):
                st.markdown("""
                **Graph Search Queries:**
                - `MATCH (n) RETURN count(n)` - Count all nodes
                - `MATCH ()-[r]->() RETURN count(r)` - Count all relationships
                - `MATCH (n) WHERE n.name CONTAINS "Alice" RETURN n` - Find nodes by name
                - `MATCH (n)-[r]->(m) RETURN n.name, type(r), m.name LIMIT 10` - Show relationships

                **Tips:**
                - Use natural language questions in the chat for semantic search
                - Direct Kuzu queries give you full control over the graph database
                - The knowledge graph combines semantic search with structured queries
                """)

    else:
        st.markdown("""
        <div style="
          min-height:580px; display:flex; flex-direction:column;
          align-items:center; justify-content:center;
          background:#1e293b; border-radius:14px;
          border:1px solid #334155; color:#475569;
        ">
          <div style="font-size:5rem; margin-bottom:1rem">🧠</div>
          <div style="font-size:1.2rem; font-weight:600; color:#64748b">
            Your knowledge graph will appear here
          </div>
          <div style="font-size:0.85rem; margin-top:0.6rem; color:#475569">
            Upload a PDF and click Build →
          </div>
          <div style="margin-top:2rem; font-size:0.78rem; color:#334155;
                      text-align:center; max-width:360px; line-height:1.6">
            Graphiti extracts entities &amp; relationships<br>
            chunk-by-chunk using Claude Sonnet,<br>
            then builds a temporal knowledge graph in KuzuDB.
          </div>
        </div>
        """, unsafe_allow_html=True)
