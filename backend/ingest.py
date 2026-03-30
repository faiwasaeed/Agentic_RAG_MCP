"""
ingest.py  – Smart section-aware ingestion pipeline
============================================================
Instead of blindly splitting by character count (which breaks
department content mid-sentence), this pipeline:

  1. Extracts all text from the PDF
  2. Detects department/section boundaries dynamically using
     numbered heading patterns (e.g. "1. Department of X", "2.3 Eligibility")
  3. Groups pages into complete department sections
  4. Creates one rich chunk per sub-section (Introduction, Programs,
     Eligibility, Faculty) so each chunk is self-contained and searchable
  5. Falls back to standard character-based splitting for any leftover text

This works on ANY prospectus-style PDF.

Run once:
    python backend/ingest.py        (from project root)
    python ingest.py                (from inside backend/)
"""

import os
import re
import sys
import time
import shutil
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
THIS_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(THIS_DIR, ".."))
PDF_PATH      = os.path.join(PROJECT_ROOT, "data", "UET_Prospectus.pdf")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
EMBED_MODEL   = "nomic-embed-text"

# ── Section heading patterns (auto-detected) ────────
# Matches patterns like:
#   "1. Department of Electrical Engineering"
#   "2.3 Eligibility Criteria:"
#   "3.4 Faculty Members:"
DEPT_HEADING_RE    = re.compile(
    r"^\s*(\d+)\.\s+(Department of|Institute of|Centre of|Center of|Automotive|"
    r"Institute of Business)",
    re.IGNORECASE | re.MULTILINE,
)
SUBSECTION_RE      = re.compile(
    r"^\s*(\d+\.\d+)\s+(Introduction|Offered Programs|Eligibility Criteria|"
    r"Faculty Members|Overview|About)[\s:]*",
    re.IGNORECASE | re.MULTILINE,
)
# Detects TOC pages: lines that look like "Section title .... page_number"
TOC_LINE_RE        = re.compile(r"\.{4,}\s*\d+\s*$", re.MULTILINE)

# ── Section type keywords (used for metadata, not for splitting) ───────────────
SECTION_TYPE_MAP = {
    "faculty":    ["chairperson", "dean", "professor", "dr.", "faculty",
                   "associate professor", "assistant professor", "lecturer"],
    "programs":   ["ph.d", "m.sc", "msc", "m.phil", "mba", "bachelor",
                   "offered program", "degree program"],
    "eligibility":["eligibility", "admission", "requirement", "criteria",
                   "sixteen-year", "18-year", "16 year"],
    "introduction":["established", "introduction", "history", "overview",
                    "founded", "distinction"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalise whitespace and strip non-ASCII control chars."""
    text = re.sub(r"[ \t]+", " ", text)          # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)        # max 2 consecutive newlines
    text = re.sub(r"[^\x20-\x7E\n]", "", text)   # strip non-printable
    return text.strip()


def is_toc_page(text: str) -> bool:
    """Return True if the page looks like a Table of Contents."""
    toc_lines = len(TOC_LINE_RE.findall(text))
    total_lines = max(len(text.splitlines()), 1)
    return (toc_lines / total_lines) > 0.25   # >25% of lines are TOC-style


def detect_section_type(text: str) -> str:
    t = text.lower()
    for stype, keywords in SECTION_TYPE_MAP.items():
        if any(kw in t for kw in keywords):
            return stype
    return "general"


def extract_department_name(heading_line: str) -> str:
    """
    Extract just the department name from a heading.
    e.g. "1. Department of Electrical Engineering" → "Electrical Engineering"
    """
    # Remove leading number+dot
    text = re.sub(r"^\s*\d+\.\s*", "", heading_line).strip()
    # Remove prefix words
    for prefix in ["Department of ", "Institute of ", "Centre of Excellence in ",
                   "Center of Excellence in "]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Core: section-aware splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sections(full_text: str) -> list[dict]:
    """
    Dynamically detect department and sub-section boundaries in the full
    PDF text. Returns a list of dicts:
        { "department": str, "section_type": str, "content": str }
    Nothing is hardcoded — boundaries are found by regex on the actual text.
    """
    lines        = full_text.splitlines()
    sections     = []
    current_dept = "General"
    current_buf  : list[str] = []
    current_type = "general"

    def flush(dept, stype, buf):
        text = clean_text("\n".join(buf))
        if len(text) > 80:   # only keep substantive chunks
            sections.append({
                "department":   dept,
                "section_type": stype,
                "content":      text,
            })

    for line in lines:
        # ── Detect top-level department heading ───────────────────────────────
        dept_match = DEPT_HEADING_RE.match(line)
        if dept_match:
            flush(current_dept, current_type, current_buf)
            current_dept = extract_department_name(line)
            current_buf  = [line]
            current_type = "introduction"
            continue

        # ── Detect sub-section heading (e.g. "2.3 Eligibility Criteria:") ────
        sub_match = SUBSECTION_RE.match(line)
        if sub_match:
            flush(current_dept, current_type, current_buf)
            sub_title    = sub_match.group(2).lower()
            current_type = detect_section_type(sub_title)
            current_buf  = [line]
            continue

        current_buf.append(line)

    # Flush last section
    flush(current_dept, current_type, current_buf)
    return sections


def fallback_split(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Simple character-based fallback for text that has no heading structure."""
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c for c in chunks if len(c.strip()) > 80]


# ─────────────────────────────────────────────────────────────────────────────
# Main ingestion
# ─────────────────────────────────────────────────────────────────────────────

def ingest():
    log.info("🚀  Starting smart section-aware ingestion …")
    log.info(f"    PDF           : {PDF_PATH}")
    log.info(f"    Vector DB     : {VECTOR_DB_DIR}")
    t0 = time.perf_counter()

    # ── Pre-flight ────────────────────────────────────────────────────────────
    if not os.path.exists(PDF_PATH):
        log.error(f"❌  PDF not found at {PDF_PATH}")
        log.error(f"    Place your PDF at: {PDF_PATH}")
        sys.exit(1)
    log.info(f"✅  PDF found  ({round(os.path.getsize(PDF_PATH)/1024/1024,1)} MB)")

    try:
        import ollama as _ol
        _ol.list()
        log.info("✅  Ollama is running")
    except Exception as e:
        log.error(f"❌  Ollama not reachable: {e}")
        log.error("    Start it with: ollama serve")
        sys.exit(1)

    # ── 1. Load all pages ─────────────────────────────────────────────────────
    from langchain_community.document_loaders import PyPDFLoader
    log.info("📄  Loading PDF pages …")
    pages = PyPDFLoader(PDF_PATH).load()
    log.info(f"    {len(pages)} pages loaded")
    if not pages:
        log.error("❌  0 pages — PDF may be image-based (scanned)")
        sys.exit(1)

    # ── 2. Filter out TOC pages, concatenate remaining text ───────────────────
    content_pages = []
    skipped_toc   = 0
    for page in pages:
        if is_toc_page(page.page_content):
            skipped_toc += 1
        else:
            content_pages.append(page)

    log.info(f"    Skipped {skipped_toc} TOC pages, kept {len(content_pages)} content pages")

    full_text = "\n".join(p.page_content for p in content_pages)
    log.info(f"    Total content text: {len(full_text):,} characters")

    # ── 3. Section-aware splitting ────────────────────────────────────────────
    log.info("✂️   Splitting into semantic sections …")
    sections = split_into_sections(full_text)
    log.info(f"    Found {len(sections)} sections from heading detection")

    # If heading detection found very few sections, fall back to char splitting
    if len(sections) < 5:
        log.warning("    ⚠️  Few sections detected — using fallback character splitting")
        fallback_texts = fallback_split(full_text)
        sections = [
            {
                "department":   "General",
                "section_type": detect_section_type(t),
                "content":      t,
            }
            for t in fallback_texts
        ]
        log.info(f"    Fallback produced {len(sections)} chunks")

    # ── 4. Convert sections → LangChain Documents ────────────────────────────
    from langchain_core.documents import Document
    docs = []
    for sec in sections:
        docs.append(Document(
            page_content = sec["content"],
            metadata     = {
                "department":   sec["department"],
                "section_type": sec["section_type"],
                "source":       "UET_Prospectus.pdf",
            }
        ))

    log.info(f"📦  Created {len(docs)} document chunks")

    # Show sample to verify quality
    log.info("    Sample chunk preview:")
    if docs:
        preview = docs[0].page_content[:200].replace("\n", " ")
        log.info(f"    [{docs[0].metadata['department']} | {docs[0].metadata['section_type']}]")
        log.info(f"    {preview} …")

    # ── 5. Embed & store ──────────────────────────────────────────────────────
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma

    log.info("⏳  Testing embedding model …")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    try:
        vec = embeddings.embed_query("test query")
        log.info(f"    Embedding OK (dim={len(vec)})")
    except Exception as e:
        log.error(f"❌  Embedding failed: {e}")
        log.error("    Run: ollama pull nomic-embed-text")
        sys.exit(1)

    # Wipe old DB
    if os.path.exists(VECTOR_DB_DIR):
        log.info(f"    Removing old vector DB …")
        shutil.rmtree(VECTOR_DB_DIR)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    log.info("⏳  Generating embeddings and storing in ChromaDB …")
    BATCH = 50
    db    = None
    for i in range(0, len(docs), BATCH):
        batch = docs[i : i + BATCH]
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=VECTOR_DB_DIR,
            )
        else:
            db.add_documents(batch)
        log.info(f"    Stored {min(i+BATCH, len(docs))}/{len(docs)} chunks …")

    elapsed = round(time.perf_counter() - t0, 1)
    count   = db._collection.count()

    log.info(f"✅  Ingestion complete!")
    log.info(f"    Chunks stored : {count}")
    log.info(f"    Time taken    : {elapsed}s")

    if count == 0:
        log.error("❌  0 chunks stored — check the PDF for extractable text")
    else:
        # Print department distribution
        dept_counts: dict[str, int] = {}
        for doc in docs:
            d = doc.metadata["department"]
            dept_counts[d] = dept_counts.get(d, 0) + 1
        log.info("    Chunks per department:")
        for dept, cnt in sorted(dept_counts.items(), key=lambda x: -x[1]):
            log.info(f"      {dept:<45} {cnt}")
        log.info("🎉  Ready! Run: python backend/main.py")


if __name__ == "__main__":
    ingest()
