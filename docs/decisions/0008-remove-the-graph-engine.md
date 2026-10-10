# ADR 0008: Remove the graph engine; Kognita holds governance records only

**Status:** Accepted (after 0.3.0)
**Supersedes:** ADR 0001 (Kuzu co-tenancy), ADR 0002 (SoR mirror); amends ADR 0003
**Closes:** ROADMAP 0.4 item 2, "The Graph Extra: Decide Its Fate"

## Context

Since 0.2, `kognita.graph` wrapped Graphiti and Kuzu into a bi-temporal knowledge graph behind the `[graph]` extra. ADR 0003 already moved it out of the top-level namespace so that `import kognita` read as a decision engine, not a GraphRAG library. The roadmap then left its fate open: test and unpin it, or extract it.

Three things settled it:

- **Scope.** Kognita is a control plane for AI governance. It holds policies, guidelines and governance records, and decides and evidences access to everything else. A knowledge graph over documents is data-plane infrastructure, the thing being governed rather than the governor.
- **Cost.** The extra pinned `graphiti-core==0.28.2` and `kuzu==0.11.3`, and capped `openai` below 2 because that Graphiti release breaks against newer OpenAI SDKs. The `[anthropic]`, `[groq]` and `[gemini]` extras installed nothing but Graphiti. None of it was used by the decision engine, the gateways or the evidence plane.
- **Fit.** Temporal governance does not need a graph. Policies are effective-dated rows replayed with `as_of`; evidence pins policy, retrieval, model and tool hashes. Freshness, supersession and review dates belong on the governance records themselves (ROADMAP 0.4 item 8; 0.6 items 2, 4 and 8).

## Decision

- `kognita.graph`, the `[graph]` extra, and the `[anthropic]`, `[groq]` and `[gemini]` extras are removed from the package. The Streamlit graph demo under `examples/` goes with them.
- No replacement package is published from this repository. The code remains in the git history and in the 0.3.0 release on PyPI.
- Kognita indexes and retrieves only policies, guidelines and governance records. Client data, document stores, agent memory, knowledge graphs and file-store connectors stay in the bank's systems; Kognita governs access to them through `decide()`, the AI gateway and the MCP proxy.
- `graphiti_core` and `kuzu` stay on the core's forbidden-import list, so neither can return to the engine unnoticed.
- Touching a removed graph name on `kognita` (`Kognita`, `GraphEngine`, `KuzuSession`, …) raises an `AttributeError` that names this ADR and how to keep the old engine, keeping ADR 0003's promise that a clean break stays navigable.

## Consequences

- `pip install kognita[graph]` on a later release warns that the extra does not exist and installs the core alone. Anyone who needs the engine pins `kognita[graph]==0.3.0`.
- `kognita doctor` no longer reports graph, Anthropic, Groq or Gemini packages.
- The `openai` extra keeps its `<2` cap for now. The cap was introduced for Graphiti; lifting it is a separate change that needs the OpenAI embedder adapter tested against the newer SDK.
- `LLMConfig` and `list_models` stay. They are standard-library only and public API, though the graph engine was their main consumer.
- Retrieval keeps its pack-supplied `subgraph` hook for subject context. It is a callback into the domain pack, not a graph database.
