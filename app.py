"""
PDF → Knowledge Graph  (Graphiti + KuzuDB + Streamlit)
-------------------------------------------------------
Upload a PDF → chunked into episodes → Graphiti extracts entities &
relationships via Claude Sonnet → interactive pyvis graph + Q&A search.
"""

import asyncio
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# ── Graphiti ──────────────────────────────────────────────────────────────────
from graphiti_core import Graphiti
from graphiti_core.driver.kuzu_driver import KuzuDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.nodes import EntityNode, EpisodeType

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


def make_graphiti(ant_key: str, oai_key: str, db_path: str) -> Graphiti:
    return Graphiti(
        driver=KuzuDriver(db=db_path),
        llm_client=AnthropicClient(
            config=LLMConfig(
                api_key=ant_key,
                model="claude-sonnet-4-20250514",
            )
        ),
        embedder=OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=oai_key,
                embedding_model="text-embedding-3-small",
                embedding_dim=1536,
            )
        ),
    )


async def ingest_pdf(
    chunks: list[str],
    pdf_name: str,
    ant_key: str,
    oai_key: str,
    db_path: str,
    progress_cb,
    status_cb,
) -> tuple[dict[str, EntityNode], list[EntityEdge], list[dict]]:
    graphiti = make_graphiti(ant_key, oai_key, db_path)
    await graphiti.build_indices_and_constraints()

    all_nodes: dict[str, EntityNode] = {}
    all_edges: list[EntityEdge] = []
    episodes_log: list[dict] = []

    n = len(chunks)
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
            episodes_log.append({
                "chunk":   idx + 1,
                "preview": chunk[:130].replace("\n", " ") + "…",
                "error":   str(exc),
                "nodes":   0,
                "edges":   0,
            })
        progress_cb((idx + 1) / n)

    await graphiti.close()
    return all_nodes, all_edges, episodes_log


async def search_graph(
    query: str, ant_key: str, oai_key: str, db_path: str
) -> list:
    graphiti = make_graphiti(ant_key, oai_key, db_path)
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


# ═══════════════════════════════════════════════════════════════════════════════
# Two-column layout
# ═══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 2], gap="large")

# ─── LEFT ────────────────────────────────────────────────────────────────────
with left:
    st.markdown("### 1 · Upload PDF")
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded:
        pdf_bytes = uploaded.read()
        st.success(f"✅ **{uploaded.name}** · {len(pdf_bytes) // 1024} KB")

        raw_text = extract_text(pdf_bytes)
        chunks   = chunk_text(raw_text, chunk_size, chunk_overlap)

        with st.expander(f"Preview extracted text · {len(raw_text):,} chars"):
            st.text_area(
                "", value=raw_text[:3000] + "\n…",
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

            # Reset
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v
            st.session_state.db_path  = db_path
            st.session_state.pdf_name = uploaded.name

            prog = st.progress(0.0)
            stat = st.empty()

            with st.spinner("Graphiti is working …"):
                nodes, edges, ep_log = run_async(
                    ingest_pdf(
                        chunks, uploaded.name,
                        anthropic_key, openai_key, db_path,
                        lambda v: prog.progress(v),
                        lambda m: stat.info(m),
                    )
                )

            st.session_state.all_nodes    = nodes
            st.session_state.all_edges    = edges
            st.session_state.episodes_log = ep_log
            st.session_state.graph_built  = True
            stat.success("✅ Graph built successfully!")

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
                    results = run_async(
                        search_graph(
                            query, anthropic_key, openai_key,
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

        tab_g, tab_l, tab_f, tab_e = st.tabs(
            ["🕸️  Graph", "📋  Episode Log", "📜  All Facts", "📥  Export"]
        )

        # Graph
        with tab_g:
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
        with tab_l:
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
        with tab_f:
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
        with tab_e:
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
