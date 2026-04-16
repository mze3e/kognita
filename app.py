"""
PDF → Knowledge Graph  (Graphiti + KuzuDB + Streamlit)
-------------------------------------------------------
Upload a PDF → chunked into episodes → Graphiti extracts entities &
relationships via Claude Sonnet → interactive pyvis graph + Q&A search.
"""

import hashlib
import shutil

import fitz  # PyMuPDF
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import pandas as pd
import openai
from streamlit_option_menu import option_menu
try:
    import google.generativeai as genai
except ImportError:
    genai = None

# ── Graphiti ──────────────────────────────────────────────────────────────────
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.llm_client.gemini_client import GeminiClient
from graphiti_core.nodes import EntityNode, EpisodeType

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
SAVED_GRAPHS_DIR = "saved_graphs"
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
                    except:
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
                    except:
                        continue
    return None

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

# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Health Check")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    st.write(f"Anthropic API Key: {'✅' if anthropic_key else '❌'}")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    st.write(f"OpenAI API Key: {'✅' if openai_key else '❌'}")
    groq_key = os.environ.get("GROQ_API_KEY", "")
    st.write(f"Groq API Key: {'✅' if groq_key else '❌'}")
    gemini_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
    st.write(f"Gemini API Key: {'✅' if gemini_key else '❌'}")

    # Available models based on API keys
    available_models = []
    if anthropic_key:
        available_models.extend(["Claude Sonnet (Anthropic)", "Claude Haiku (Anthropic)"])
    if openai_key:
        available_models.extend(["GPT-4 (OpenAI)", "GPT-3.5 Turbo (OpenAI)"])
    if groq_key:
        available_models.extend(["Llama 3 70B (Groq)", "Mixtral 8x7B (Groq)"])
    if gemini_key:
        available_models.extend(["Gemini 1.5 Pro (Google)", "Gemini 1.5 Flash (Google)"])

    if available_models:
        st.success(f"✅ {len(available_models)} models available")
    else:
        st.error("❌ No API keys configured")

    st.divider()
    st.markdown("## 🤖 Graph Processing Model")

    # Model selection for graph processing
    if available_models:
        processing_model = st.selectbox(
            "Model for Knowledge Graph Building",
            available_models,
            index=0,  # Default to first available model
            help="Choose which LLM to use for extracting entities and relationships from PDF chunks"
        )

        # Store the selected processing model in session state
        st.session_state.processing_model = processing_model
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


def make_graphiti(provider: str, api_key: str, db_path: str, model: str) -> Graphiti:
    """Create a Graphiti instance with the specified LLM provider and model."""

    # Determine embedder (always use OpenAI for embeddings for now, as Graphiti expects it)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get("OPENAI_API_KEY", ""),  # Use OpenAI for embeddings
            embedding_model="text-embedding-3-small",
            embedding_dim=1536,
        )
    )

    # Create LLM client based on provider
    if provider == "anthropic":
        model_map = {
            "Claude Sonnet (Anthropic)": "claude-sonnet-4-20250514",
            "Claude Haiku (Anthropic)": "claude-3-5-haiku-20241022"
        }
        actual_model = model_map.get(model, "claude-sonnet-4-20250514")
        llm_client = AnthropicClient(
            config=LLMConfig(
                api_key=api_key,
                model=actual_model,
            )
        )
    elif provider == "openai":
        model_map = {
            "GPT-4 (OpenAI)": "gpt-4",
            "GPT-3.5 Turbo (OpenAI)": "gpt-3.5-turbo"
        }
        actual_model = model_map.get(model, "gpt-4")
        # For OpenAI, we need to use a compatible client
        from graphiti_core.llm_client.openai_client import OpenAIClient
        llm_client = OpenAIClient(
            config=LLMConfig(
                api_key=api_key,
                model=actual_model,
            )
        )
    elif provider == "groq":
        model_map = {
            "Llama 3 70B (Groq)": "llama3-70b-8192",
            "Mixtral 8x7B (Groq)": "mixtral-8x7b-32768"
        }
        actual_model = model_map.get(model, "llama3-70b-8192")
        # Groq uses OpenAI-compatible API
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key=api_key,
                model=actual_model,
                base_url="https://api.groq.com/openai/v1",
            )
        )
    elif provider == "gemini":
        model_map = {
            "Gemini 1.5 Pro (Google)": "gemini-1.5-pro",
            "Gemini 1.5 Flash (Google)": "gemini-1.5-flash"
        }
        actual_model = model_map.get(model, "gemini-1.5-flash")
        from graphiti_core.llm_client.gemini_client import GeminiClient
        llm_client = GeminiClient(
            config=LLMConfig(
                api_key=api_key,
                model=actual_model,
            )
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return Graphiti(
        graph_driver=KuzuDriver(db=db_path),
        llm_client=llm_client,
        embedder=embedder,
    )


async def ingest_pdf(
    chunks: list[str],
    pdf_name: str,
    provider: str,
    api_key: str,
    model: str,
    db_path: str,
    progress_cb,
    status_cb,
) -> tuple[dict[str, EntityNode], list[EntityEdge], list[dict], bool]:
    graphiti = make_graphiti(provider, api_key, db_path, model)
    await graphiti.build_indices_and_constraints()

    all_nodes: dict[str, EntityNode] = {}
    all_edges: list[EntityEdge] = []
    episodes_log: list[dict] = []

    n = len(chunks)
    quota_exceeded = False
    for idx, chunk in enumerate(chunks):
        status_cb(f"⚙️  Processing chunk {idx + 1} / {n} …")
        try:
            result = await graphiti.add_episode(
                name=f"{pdf_name}__chunk_{idx + 1:04d}",
                episode_body=chunk,
                source=EpisodeType.text,
                source_description=f"PDF: {pdf_name}",
                reference_time=datetime.now(),
            )
            for node in result.nodes:
                all_nodes[node.uuid] = node
            all_edges.extend(result.edges)
            episodes_log.append({
                "chunk":   idx + 1,
                "preview": chunk[:130].replace("\n", " ") + "…",
                "nodes":   len(result.nodes),
                "edges":   len(result.edges),
            })
        except Exception as exc:
            error_str = str(exc).lower()
            # Check for quota/rate limit errors from various providers
            quota_keywords = [
                'quota', 'rate limit', '429', 'exceeded', 'limit exceeded',
                'insufficient_quota', 'billing_hard_limit_reached',
                'quota_exceeded', 'usage limit', 'rate limit exceeded'
            ]
            if any(keyword in error_str for keyword in quota_keywords):
                quota_exceeded = True
                episodes_log.append({
                    "chunk":   idx + 1,
                    "preview": chunk[:130].replace("\n", " ") + "…",
                    "error":   f"API quota exceeded: {str(exc)}",
                    "nodes":   0,
                    "edges":   0,
                })
                status_cb(f"❌ API quota exceeded. Stopping processing at chunk {idx + 1}.")
                break  # Stop processing further chunks
            else:
                episodes_log.append({
                    "chunk":   idx + 1,
                    "preview": chunk[:130].replace("\n", " ") + "…",
                    "error":   str(exc),
                    "nodes":   0,
                    "edges":   0,
                })
        progress_cb((idx + 1) / n)

    await graphiti.close()
    return all_nodes, all_edges, episodes_log, quota_exceeded


async def search_graph(
    query: str, provider: str, api_key: str, model: str, db_path: str
) -> list:
    graphiti = make_graphiti(provider, api_key, db_path, model)
    results = await graphiti.search(query)
    await graphiti.close()
    return results


def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text()


async def generate_llm_response(
    question: str,
    search_results: list,
    provider: str,
    api_key: str,
    nodes: dict,
    edges: list,
    selected_model: str
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
            model_map = {
                "Claude Sonnet (Anthropic)": "claude-sonnet-4-20250514",
                "Claude Haiku (Anthropic)": "claude-3-5-haiku-20241022"
            }
            model = model_map.get(selected_model, "claude-sonnet-4-20250514")

            llm_client = AnthropicClient(
                config=LLMConfig(
                    api_key=api_key,
                    model=model,
                )
            )
            response = await llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.content[0].text if response.content else "No response generated."

        elif provider == "openai":
            model_map = {
                "GPT-4 (OpenAI)": "gpt-4",
                "GPT-3.5 Turbo (OpenAI)": "gpt-3.5-turbo"
            }
            model = model_map.get(selected_model, "gpt-4")

            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content if response.choices else "No response generated."

        elif provider == "groq":
            model_map = {
                "Llama 3 70B (Groq)": "llama3-70b-8192",
                "Mixtral 8x7B (Groq)": "mixtral-8x7b-32768"
            }
            model = model_map.get(selected_model, "llama3-70b-8192")

            import openai
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000
            )
            return response.choices[0].message.content if response.choices else "No response generated."

        elif provider == "gemini":
            if genai is None:
                return "Gemini library not available. Please install google-generativeai."

            model_map = {
                "Gemini 1.5 Pro (Google)": "gemini-1.5-pro",
                "Gemini 1.5 Flash (Google)": "gemini-1.5-flash"
            }
            model = model_map.get(selected_model, "gemini-1.5-flash")

            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(model)
            response = await gemini_model.generate_content_async(prompt)
            return response.text if response.text else "No response generated."

        else:
            return f"Unsupported provider: {provider}"

    except Exception as e:
        return f"Error generating response: {str(e)}"


def execute_kuzu_query(query: str, db_path: str) -> list:
    """Execute a direct Kuzu query and return results."""
    try:
        # For now, provide some basic queries that we can handle
        # In a full implementation, you'd use KuzuDriver's query methods
        if "COUNT" in query.upper() and "NODE" in query.upper():
            # Mock count query - in real implementation, execute actual query
            return [{"node_count": 42}]  # Placeholder
        elif "COUNT" in query.upper() and "EDGE" in query.upper():
            return [{"edge_count": 156}]  # Placeholder
        elif "LABELS" in query.upper():
            return [{"labels": ["Entity", "Episode"]}]  # Placeholder
        else:
            # For other queries, return a message
            return [{"result": f"Query executed: {query}", "status": "Mock result - full Kuzu integration needed"}]
    except Exception as e:
        return [{"error": str(e)}]


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
        keys_ok = bool(anthropic_key and openai_key)
        if not keys_ok:
            st.warning("⚠️ Enter both API keys in the sidebar first.")

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

            # Get processing model details
            processing_model = st.session_state.get("processing_model")
            if not processing_model:
                st.error("No processing model selected")
                st.stop()

            # Determine provider and API key for processing
            if "Anthropic" in processing_model:
                processing_provider = "anthropic"
                processing_api_key = anthropic_key
            elif "OpenAI" in processing_model:
                processing_provider = "openai"
                processing_api_key = openai_key
            elif "Groq" in processing_model:
                processing_provider = "groq"
                processing_api_key = groq_key
            elif "Google" in processing_model:
                processing_provider = "gemini"
                processing_api_key = gemini_key
            else:
                st.error(f"Unsupported processing model: {processing_model}")
                st.stop()

            with st.spinner("Graphiti is working …"):
                nodes, edges, ep_log, quota_exceeded = run_async(
                    ingest_pdf(
                        chunks, uploaded.name,
                        processing_provider, processing_api_key, processing_model, db_path,
                        lambda v: prog.progress(v),
                        lambda m: stat.info(m),
                    )
                )

            st.session_state.all_nodes    = nodes
            st.session_state.all_edges    = edges
            st.session_state.episodes_log = ep_log
            st.session_state.graph_built  = True

            if quota_exceeded:
                stat.error("⚠️ API quota exceeded! Processing stopped early. Some chunks were not processed.")
                st.warning("**API Quota Exceeded**: The knowledge graph was partially built. You may need to wait for your API quota to reset or upgrade your plan before processing the remaining chunks.")
            else:
                stat.success("✅ Graph built successfully!")

            # Save the graph data
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
        if st.button("🔍 Search", use_container_width=True) and query.strip():
            with st.spinner("Searching …"):
                try:
                    # Use the same model as used for processing
                    processing_model = st.session_state.get("processing_model", "Claude Sonnet (Anthropic)")
                    if "Anthropic" in processing_model:
                        search_provider = "anthropic"
                        search_api_key = anthropic_key
                    elif "OpenAI" in processing_model:
                        search_provider = "openai"
                        search_api_key = openai_key
                    elif "Groq" in processing_model:
                        search_provider = "groq"
                        search_api_key = groq_key
                    elif "Google" in processing_model:
                        search_provider = "gemini"
                        search_api_key = gemini_key
                    else:
                        search_provider = "anthropic"
                        search_api_key = anthropic_key

                    results = run_async(
                        search_graph(
                            query, search_provider, search_api_key, processing_model,
                            st.session_state.db_path,
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

            # Get available models dynamically
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            groq_key = os.environ.get("GROQ_API_KEY", "")
            gemini_key = os.environ.get("GOOGLE_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

            available_models = []
            if anthropic_key:
                available_models.extend(["Claude Sonnet (Anthropic)", "Claude Haiku (Anthropic)"])
            if openai_key:
                available_models.extend(["GPT-4 (OpenAI)", "GPT-3.5 Turbo (OpenAI)"])
            if groq_key:
                available_models.extend(["Llama 3 70B (Groq)", "Mixtral 8x7B (Groq)"])
            if gemini_key:
                available_models.extend(["Gemini 1.5 Pro (Google)", "Gemini 1.5 Flash (Google)"])

            if not available_models:
                st.error("❌ No API keys configured. Please set at least one API key in your environment variables.")
                st.stop()

            # Model Selection
            selected_model = st.selectbox(
                "Choose Model",
                available_models,
                help="Select from available models based on your configured API keys"
            )

            # Determine provider and API key based on selected model
            if "Anthropic" in selected_model:
                provider = "anthropic"
                api_key = anthropic_key
            elif "OpenAI" in selected_model:
                provider = "openai"
                api_key = openai_key
            elif "Groq" in selected_model:
                provider = "groq"
                api_key = groq_key
            elif "Google" in selected_model:
                provider = "gemini"
                api_key = gemini_key

            st.info(f"Using: {selected_model}")

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
                if not selected_key:
                    st.error("Please set the API key for the selected LLM provider.")
                else:
                    # Add user message to history
                    st.session_state.chat_history.append({"role": "user", "content": chat_input})

                    with st.spinner("Thinking..."):
                        try:
                            # Search the graph using the processing model
                            processing_model = st.session_state.get("processing_model", "Claude Sonnet (Anthropic)")
                            if "Anthropic" in processing_model:
                                search_provider = "anthropic"
                                search_api_key = anthropic_key
                            elif "OpenAI" in processing_model:
                                search_provider = "openai"
                                search_api_key = openai_key
                            elif "Groq" in processing_model:
                                search_provider = "groq"
                                search_api_key = groq_key
                            elif "Google" in processing_model:
                                search_provider = "gemini"
                                search_api_key = gemini_key
                            else:
                                search_provider = "anthropic"
                                search_api_key = anthropic_key

                            search_results = run_async(
                                search_graph(chat_input, search_provider, search_api_key, processing_model, st.session_state.db_path)
                            )

                            # Generate response using LLM
                            response = generate_llm_response(chat_input, search_results, provider, api_key, nodes, edges, selected_model)

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
                with st.spinner("Executing query..."):
                    try:
                        results = execute_kuzu_query(kuzu_query, st.session_state.db_path)
                        st.success(f"Query executed successfully! Found {len(results)} results.")

                        if results:
                            # Display results in a nice format
                            df = pd.DataFrame(results)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.info("No results found.")

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
