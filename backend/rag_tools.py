"""
rag_tools.py - Accurate RAG pipeline exposed as MCP-callable tools

No hardcoded data. Everything comes from the vector DB built by ingest.py.
The key to accuracy is the smarter chunking in ingest.py (section-aware),
combined with multi-query expansion and reranking here.
"""

import os
import re
import time
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM as Ollama

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, ".."))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
EMBED_MODEL   = "nomic-embed-text"
LLM_MODEL     = "gemma3"
MMR_K         = 5
MMR_FETCH_K   = 20
TOP_K_DOCS    = 6
TEMPERATURE   = 0

# Keywords that indicate a query is UET-related (used for scope guardrail)
UET_KEYWORDS = [
    "uet", "university of engineering", "department", "faculty", "admission",
    "program", "degree", "course", "semester", "fee", "hostel", "lab",
    "curriculum", "prospectus", "chairperson", "chairman", "dean", "professor",
    "eligibility", "phd", "msc", "mphil", "mba", "bs", "ms",
    "electrical", "computer", "mechanical", "civil", "chemical", "software",
    "architecture", "mathematics", "physics", "chemistry", "mining",
    "petroleum", "geological", "metallurgical", "polymer", "automotive",
    "mechatronics", "industrial", "transportation", "environmental",
    "water resources", "data science", "islamic", "business", "management",
]

# ── Singletons ────────────────────────────────────────────────────────────────
_embeddings : Optional[OllamaEmbeddings] = None
_vector_db  : Optional[Chroma]           = None
_llm        : Optional[Ollama]           = None


def _load_resources():
    """Load embedding model, vector DB, and LLM once at startup."""
    global _embeddings, _vector_db, _llm
    if _embeddings is None:
        log.info("[RAG] Loading embedding model ...")
        _embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    if _vector_db is None:
        abs_path = os.path.abspath(VECTOR_DB_DIR)
        log.info(f"[RAG] Loading ChromaDB from: {abs_path}")
        if not os.path.exists(abs_path):
            raise RuntimeError(
                f"Vector DB not found at {abs_path}. "
                "Run `python backend/ingest.py` first."
            )
        _vector_db = Chroma(
            persist_directory=abs_path,
            embedding_function=_embeddings,
        )
        count = _vector_db._collection.count()
        log.info(f"[RAG] Vector DB ready - {count} chunks indexed")
        if count == 0:
            log.warning("[RAG] WARNING: 0 chunks in DB. Re-run ingest.py.")
    if _llm is None:
        log.info(f"[RAG] Loading LLM ({LLM_MODEL}) ...")
        _llm = Ollama(model=LLM_MODEL, temperature=TEMPERATURE)
    return _embeddings, _vector_db, _llm


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(query: str) -> list[str]:
    """
    Generate up to 4 query variants to improve recall.
    Completely dynamic - analyses the query itself, no hardcoded dept names.
    """
    q       = query.strip()
    q_lower = q.lower()
    variants = [q]

    # Extract a potential department name from the query
    # Look for "of X Engineering/Science/etc" pattern
    dept_match = re.search(
        r"(?:of|in|for|about)\s+([A-Z][a-zA-Z\s&]+?(?:Engineering|Science|"
        r"Planning|Architecture|Management|Studies|Technology|Design|Mathematics|"
        r"Physics|Chemistry|Mining|Geology|Petroleum))",
        query, re.IGNORECASE
    )
    dept_hint = dept_match.group(1).strip() if dept_match else ""

    # Intent-based expansions
    if any(w in q_lower for w in ["chairperson", "chairman", "chair", "head"]):
        variants.append(f"chairperson chairman head {dept_hint}".strip())
        variants.append(f"faculty members professors {dept_hint}".strip())

    elif any(w in q_lower for w in ["dean"]):
        variants.append(f"dean faculty {dept_hint}".strip())

    elif any(w in q_lower for w in ["faculty", "professor", "staff", "lecturer"]):
        variants.append(f"professors associate assistant faculty {dept_hint}".strip())
        variants.append(f"faculty members staff {dept_hint}".strip())

    elif any(w in q_lower for w in ["program", "degree", "offer", "course",
                                     "phd", "msc", "bs", "ms", "mphil", "mba"]):
        variants.append(f"offered programs degrees {dept_hint}".strip())
        variants.append(f"PhD MSc MPhil bachelor programs {dept_hint}".strip())

    elif any(w in q_lower for w in ["eligib", "admission", "require", "criteria",
                                     "qualify", "entry", "apply"]):
        variants.append(f"eligibility criteria admission requirements {dept_hint}".strip())
        variants.append(f"sixteen year education degree {dept_hint}".strip())

    elif any(w in q_lower for w in ["introduc", "about", "history", "overview",
                                     "established", "tell me", "what is"]):
        variants.append(f"introduction history established {dept_hint}".strip())
        variants.append(dept_hint if dept_hint else q)

    else:
        # Generic fallback: add department hint as standalone
        if dept_hint:
            variants.append(dept_hint)

    # Deduplicate while preserving order
    seen   : set[str] = set()
    result : list[str] = []
    for v in variants:
        v_clean = v.strip()
        if v_clean and v_clean.lower() not in seen:
            seen.add(v_clean.lower())
            result.append(v_clean)
    return result[:4]


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank(query: str, docs: list) -> list:
    """
    Score documents by relevance to the query.
    Prioritises exact phrase matches, keyword overlap, and
    faculty/leadership terms when relevant.
    """
    q_lower  = query.lower()
    q_words  = set(re.findall(r"\b\w{4,}\b", q_lower))
    is_faculty_query = any(w in q_lower for w in
                           ["chairperson", "chairman", "dean", "director",
                            "faculty", "professor", "lecturer"])
    scored = []
    for doc in docs:
        text  = doc.page_content.lower()
        score = 0

        # Exact 4-word phrase match (high signal)
        words = q_lower.split()
        for i in range(len(words) - 3):
            phrase = " ".join(words[i:i+4])
            if phrase in text:
                score += 40

        # 3-word phrase match
        for i in range(len(words) - 2):
            phrase = " ".join(words[i:i+3])
            if phrase in text:
                score += 15

        # Keyword overlap
        text_words = set(re.findall(r"\b\w{4,}\b", text))
        score += 2 * len(q_words & text_words)

        # Faculty role terms present in both query and document
        if is_faculty_query:
            for term in ["chairperson", "chairman", "dean", "director",
                         "professor", "lecturer", "dr."]:
                if term in text:
                    score += 10
                    break

        # Section type bonus: faculty queries should prefer faculty chunks
        stype = doc.metadata.get("section_type", "")
        if is_faculty_query and stype == "faculty":
            score += 20
        elif any(w in q_lower for w in ["program", "degree", "phd", "msc"]) \
             and stype == "programs":
            score += 20
        elif any(w in q_lower for w in ["eligib", "admission"]) \
             and stype == "eligibility":
            score += 20

        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored]


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Handlers
# ═══════════════════════════════════════════════════════════════════════════════

def tool_validate_query(query: str) -> dict:
    """MCP Tool: validate_query — check scope."""
    if not query.strip():
        return {"is_valid": False, "reason": "Empty query"}
    q = query.lower()
    if any(kw in q for kw in UET_KEYWORDS):
        return {"is_valid": True, "reason": "Query is UET-related"}
    return {
        "is_valid": False,
        "reason": (
            "I only answer questions about UET departments, programs, "
            "admissions, faculty, fees, and facilities."
        ),
    }


def tool_retrieve_context(query: str, department: str = "") -> dict:
    """
    MCP Tool: retrieve_context
    Multi-query expansion → MMR vector search → reranking.
    All data comes from the vector DB — nothing hardcoded.
    """
    _, db, _ = _load_resources()
    variants = expand_query(query)
    log.info(f"[RAG] Variants: {variants}")

    seen_hashes : set[int] = set()
    all_docs    : list     = []

    for variant in variants:
        search_kwargs: dict = dict(k=MMR_K, fetch_k=MMR_FETCH_K, lambda_mult=0.6)
        if department:
            search_kwargs["filter"] = {"department": department}
        try:
            docs = db.max_marginal_relevance_search(variant, **search_kwargs)
            for doc in docs:
                h = hash(doc.page_content[:300])
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    all_docs.append(doc)
        except Exception as e:
            log.warning(f"[RAG] Search error for '{variant}': {e}")

    ranked = rerank(query, all_docs)[:TOP_K_DOCS]
    log.info(f"[RAG] {len(all_docs)} unique docs retrieved → top {len(ranked)} after reranking")

    chunks = [
        {
            "content":      doc.page_content,
            "department":   doc.metadata.get("department", "Unknown"),
            "section_type": doc.metadata.get("section_type", "general"),
            "source":       doc.metadata.get("source", "UET_Prospectus.pdf"),
        }
        for doc in ranked
    ]
    return {"chunks": chunks, "num_chunks": len(chunks)}


def tool_generate_answer(query: str, context_chunks: list) -> dict:
    """MCP Tool: generate_answer — LLM synthesis over retrieved chunks."""
    if not context_chunks:
        return {
            "answer":     "No relevant information found in the UET prospectus for your question.",
            "sources":    [],
            "llm_time_s": 0,
        }

    _, _, llm = _load_resources()

    context_parts = []
    sources       = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[Chunk {i} | {chunk['department']} | {chunk['section_type']}]\n"
            f"{chunk['content']}"
        )
        sources.append(f"{chunk['department']} ({chunk['section_type']})")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert assistant for the University of Engineering and Technology (UET) Lahore.

Answer the user's question using ONLY the information provided in the context chunks below.
- Be direct and specific.
- List faculty names, programs, or criteria as bullet points when relevant.
- If the exact answer is not in the context, say so honestly.
- Do NOT make up information.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    t0      = time.perf_counter()
    answer  = llm.invoke(prompt)
    elapsed = round(time.perf_counter() - t0, 2)
    log.info(f"[RAG] LLM answered in {elapsed}s")

    return {
        "answer":     answer.strip(),
        "sources":    list(dict.fromkeys(sources)),
        "llm_time_s": elapsed,
    }


def tool_list_departments(query: str = "") -> dict:
    """MCP Tool: list_departments — dynamically read from the vector DB."""
    try:
        _, db, _ = _load_resources()
        # Pull all metadata and collect unique department names
        result = db.get(include=["metadatas"])
        depts  = sorted(set(
            m["department"]
            for m in result.get("metadatas", [])
            if m.get("department") and m["department"] != "General"
        ))
        return {
            "departments": depts,
            "total":       len(depts),
            "note":        "These departments were extracted automatically from your PDF.",
        }
    except Exception as e:
        return {"departments": [], "error": str(e)}
