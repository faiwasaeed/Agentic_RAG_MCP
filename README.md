# 🎓 UET MCP-RAG System v3.0

> **Best of both worlds**: Accurate answers (from `UET_RAG_System`) + Fast MCP-based agent architecture (from `AI_Agent_MCP`)

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-orange)
![MCP](https://img.shields.io/badge/Architecture-MCP%20Agent-purple)

---

## 🧠 Why This Project Exists

| Problem | AI_Agent_MCP | UET_RAG_System | **This Project** |
|---|---|---|---|
| Response time | ⚡ Fast (~1s) | 🐢 Slow (40–80s) | ⚡ Fast (3–8s) |
| Answer accuracy | ❌ Hardcoded / low | ✅ High (RAG) | ✅ High (RAG) |
| Architecture | ✅ MCP agent | ❌ Monolithic | ✅ MCP agent |
| Query expansion | ❌ No | ✅ Yes | ✅ Yes |
| Reranking | ❌ No | ✅ Yes | ✅ Yes |
| Scope guardrail | ✅ Yes | ✅ Yes | ✅ Yes |

### Root Cause Analysis

**AI_Agent_MCP is fast but inaccurate** because it uses a hardcoded knowledge base instead of real vector retrieval from the PDF.

**UET_RAG_System is accurate but slow** because it loads the embedding model + Chroma DB + LLM on every request (no pre-loading).

**This project fixes both**: resources are loaded once at startup (fast), and the full RAG pipeline (query expansion + MMR + reranking + LLM grounding) runs on every query (accurate).

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌───────────────────────────────────────────────────────┐
│                  Streamlit Frontend                   │
│   (Chat UI with department filter + session stats)    │
└────────────────────────┬──────────────────────────────┘
                         │ HTTP POST /chat
                         ▼
┌───────────────────────────────────────────────────────┐
│                 FastAPI Backend                       │
│          (pre-loads agent on startup)                 │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────┐
│              UET MCP Agent (Orchestrator)             │
│                                                       │
│  Step 1 ─► [MCP Tool] validate_query                  │
│               └─ scope check (UET keywords)           │
│                                                       │
│  Step 2 ─► [MCP Tool] retrieve_context                │
│               ├─ Query expansion (4 variants)         │
│               ├─ MMR search on ChromaDB               │
│               └─ Reranking (phrase + keyword + dept)  │
│                                                       │
│  Step 3 ─► [MCP Tool] generate_answer                 │
│               ├─ Grounded prompt construction         │
│               └─ Local LLM (Gemma3 via Ollama)        │
└───────────────────────────────────────────────────────┘
                         │
                         ▼
              Structured JSON Response
              (answer + sources + timing + tools used)
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| 🔧 **MCP Tool Registration** | 4 tools: `validate_query`, `retrieve_context`, `generate_answer`, `list_departments` |
| 🔍 **Multi-Query Expansion** | Generates up to 4 query variants for better recall |
| 🏆 **Reranking Algorithm** | Phrase match (+40), keyword overlap (+2), faculty terms (+15), dept match (+10) |
| ⚡ **Pre-loaded Resources** | Embedding model + ChromaDB + LLM loaded once at startup – no cold-start |
| 🛡️ **Scope Guardrail** | Validates all queries before expensive retrieval |
| 📄 **Source Attribution** | Every answer cites department + page number |
| 🌐 **Department Filter** | Optional filter to search within a specific department |
| 💬 **Streamlit UI** | Session stats, sample questions, tool call transparency |
| 🧪 **22 Test Cases** | Faculty, programs, admission, fees, facilities, tricky, out-of-scope |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- [Ollama](https://ollama.ai) installed and running
- 8GB RAM minimum

### 1. Install Ollama models

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull gemma3             # LLM for answer generation
```

### 2. Clone and install

```bash
git clone https://github.com/your-username/uet-mcp-rag.git
cd uet-mcp-rag
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add the PDF

Place the UET prospectus PDF at:
```
data/UET_Prospectus.pdf
```

### 4. Ingest (one-time)

```bash
python backend/ingest.py
```

### 5. Run everything

```bash
python run_all.py
```

Or manually:

```bash
# Terminal 1
cd backend && python main.py

# Terminal 2
streamlit run frontend/app.py
```

- **Chat UI** → http://localhost:8501  
- **API Docs** → http://localhost:8000/docs  

---

## 📁 Project Structure

```
uet_mcp_rag/
│
├── backend/
│   ├── mcp_server.py     # MCP Server: tool registry + dispatcher
│   ├── mcp_agent.py      # Agent orchestrator: calls tools in sequence
│   ├── rag_tools.py      # Tool handlers: validate, retrieve, generate, list
│   ├── ingest.py         # One-time PDF → ChromaDB pipeline
│   └── main.py           # FastAPI server
│
├── frontend/
│   └── app.py            # Streamlit chat UI
│
├── tests/
│   └── test_agent.py     # 22-case automated test suite
│
├── data/
│   ├── UET_Prospectus.pdf    ← place your PDF here
│   └── vector_db/            ← auto-generated by ingest.py
│
├── run_all.py            # Combined launcher
├── requirements.txt
└── README.md
```

---

## 🔧 MCP Tools Reference

| Tool | Input | Output |
|---|---|---|
| `validate_query` | `query: str` | `{is_valid, reason}` |
| `retrieve_context` | `query, department` | `{chunks[], num_chunks}` |
| `generate_answer` | `query, context_chunks[]` | `{answer, sources[], llm_time_s}` |
| `list_departments` | `query` | `{departments[]}` |

---

## ⚙️ Configuration

Edit `backend/rag_tools.py`:

```python
EMBED_MODEL   = "nomic-embed-text"   # Ollama embedding model
LLM_MODEL     = "gemma3"             # Ollama LLM
CHUNK_SIZE    = 800                  # PDF chunk size
CHUNK_OVERLAP = 300                  # Chunk overlap
MMR_K         = 4                    # Docs per query variant
MMR_FETCH_K   = 12                   # Candidate pool size
TOP_K_DOCS    = 6                    # Docs sent to LLM
TEMPERATURE   = 0                    # LLM determinism
```

---

## 🧪 Testing

```bash
# Make sure the API is running first
python tests/test_agent.py
```

Expected output:
```
[01] ✅ PASS |   3.2s | [faculty]
[02] ✅ PASS |   2.8s | [faculty]
...
Score: 95.5%
```

---

## 📊 Performance Expectations

| Metric | Value |
|---|---|
| Cold-start (first request) | Instant (resources pre-loaded on startup) |
| Typical response time | 3–8 seconds |
| Embedding generation | ~200ms (4 variants) |
| Vector search | ~100ms |
| Reranking | <50ms |
| LLM generation | 2–6 seconds |

---

## 🙏 Acknowledgements

- **AI_Agent_MCP** by salihamubeen – MCP architecture inspiration
- **UET_RAG_System** by faiwasaeed – Accurate RAG pipeline inspiration
- LangChain, ChromaDB, Ollama, FastAPI, Streamlit
