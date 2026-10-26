# Production RAG Engine 🧠⚡

Production Retrieval-Augmented Generation (RAG) platform featuring strict Pydantic V2 data contracts, an in-memory L2-normalized cosine vector store, semantic reranking, and hallucination guardrails.

![Architecture](https://img.shields.io/badge/Architecture-VectorDB%20%2B%20FastAPI-blue)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-success)
![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen)



## 🏗️ System Overview

The engine processes unstructured text via deterministic chunking, generates dense embeddings, performs normalized cosine similarity lookups, and scores nearest-neighbor context blocks for LLM generation:

```text
Document Stream / Raw Text
            │
            ▼
┌────────────────────────────────────────────────────────┐
│              Chunking & Schema Validator               │
│  • Pydantic V2 DocumentChunk Strict Contracts          │
│  • Metadata Tagging & Unique Chunk Identifier Hashes   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            In-Memory Vector Search Engine              │
│  • Unit L2 Normalization (Hypersphere Projection)      │
│  • Cosine Similarity Dot Products: O(N * D)            │
│  • Similarity Threshold Cutoff Filtering               │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│              Retrieved Context Payloads                │
│   { chunk_id, text, similarity_score, metadata }       │
└────────────────────────────────────────────────────────┘


```
## 📁 Repository Structure

```
rag-production-engine/
├── app/
│   ├── core/           # Reranking and guardrail algorithms
│   ├── retrieval/      # Normalized cosine vector store
│   └── schemas/        # Pydantic V2 data contracts
├── tests/              # Pytest verification suites
├── requirements.txt    # Production dependencies
└── pytest.ini          # Test runner configuration
```
## ⚡ Core Engineering Invariants

- **Unit $L_2$ Hypersphere Normalization (`app/retrieval/vector_store.py`):** Normalizes document and query vectors to unit length ($\Vert{}v\Vert{}_2 = 1$). This reduces similarity search to an $O(N \cdot D)$ matrix-vector dot product:
  $$\text{sim}(q, d_i) = q \cdot d_i$$
- **Strict Data Contracts (`app/schemas/payloads.py`):** Pydantic V2 models enforce payload bounds, non-empty chunks, bounded candidate limits ($1 \le k \le 20$), and numerical cutoff thresholds ($0.0 \le \tau \le 1.0$).
- **Automated Invariant Verification (`tests/test_vector_store.py`):** Validates deterministic nearest-neighbor ranking, score-threshold pruning, and empty-index query resilience.

## 🛠️ API & Schema Contracts

| Model | Schema Name | Responsibility | Key Validations |
| :--- | :--- | :--- | :--- |
| **Ingestion Chunk** | `DocumentChunk` | Individual indexed passage | `min_length=5`, unique `chunk_id`, metadata dictionary |
| **Ingestion Batch** | `IngestionRequest` | Bulk document ingestion payload | Validates non-empty document arrays |
| **Query Request** | `QueryRequest` | Search parameters & filters | Top-$k$ integer range $[1, 20]$, threshold float $[0.0, 1.0]$ |
| **Query Output** | `QueryResponse` | Ranked context chunks | Returns descending scores and latency profiling in ms |

## 🧪 Automated Testing & Verification

Run the test suite locally:

```powershell
pytest -v tests/


---
```

## 🚀 Phase 7: Containerization & Gateway Profiling

### Production Container Build & Execution
```bash
# Build multi-stage slim container
docker build -t rag-production-engine:latest .

# Run container with non-root privileges & health probe
docker run -d -p 8000:8000 --name rag-service rag-production-engine:latest
---
```

### High-Concurrency Gateway Stress Test
```markdown

```bash
python benchmarks/load_test.py


```
## 🛠️ Automated CI & Diagnostics Probes
- **GitHub Actions:** Matrix automated validation across Python 3.10, 3.11, and 3.12.
- **Diagnostic Probes:** \/v1/diagnostics/health\ and \/v1/diagnostics/telemetry\ for container orchestrator readiness.

---

## 📑 API Contracts & Offline Specifications
- **Offline OpenAPI Spec:** Available under \docs/specs/openapi.json\.
- **Automated Spec Export:** Run \python scripts/export_openapi.py\ on route updates.


---

## 🧠 Phase 10: Grounded Context Synthesis & Citations

- **Unified Gateway Endpoint:** \/v1/generate\ integrates document ingestion, dense vector similarity, and deterministic context synthesis.
- **Citation Provenance:** Traces citations back to ingested \chunk_id\ instances with guaranteed fallback behavior for empty or ungrounded queries.
- **Deterministic Pipeline:** Synthesizes structured \GenerationResponse\ models with latency telemetry and retrieved passage attribution.

---

## 📊 Phase 11: Production Evaluation & RAG Triad Metrics

- **Context Relevance:** Evaluates query-to-context information density and filters uninformative background noise.
- **Faithfulness (Groundedness):** Verifies that synthesized claims strictly derive from retrieved context chunks to prevent hallucination.
- **Answer Relevance:** Validates bidirectional token coverage between original query intent and synthesized responses.
- **Automated Quality Gates:** Integrated threshold scoring enforcing minimal acceptable metrics before output emission.


---

## 📡 Phase 12: Distributed Observability & OpenTelemetry Instrumentation

- **Trace Propagation:** W3C Trace Context (\	raceparent\) compliant spanning across gateway lifecycles.
- **Hierarchical Profiling:** Granular microsecond span telemetry for chunk retrieval, cross-encoder rerank, and synthesis.
- **Resilience Telemetry:** Explicit span error-boundary capture with serialized exception metadata.


- [x] **Phase 14**: Dynamic Query Expansion & Reciprocal Rank Fusion (Complete)

### Phase 14: Dynamic Query Expansion & Reciprocal Rank Fusion
- **Perspective Diversification**: Sub-query generator deconstructs prompts into technical and conceptual retrieval variations.
- **Asynchronous Fan-out**: Queries vector collections concurrently via syncio.gather.
- **RRF Aggregation**: Combines multi-perspective rankings using reciprocal rank weighting (=60$) to favor persistent consensus.
