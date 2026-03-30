"""
app.py – Streamlit chat interface for UET MCP-RAG system
"""

import uuid
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

SAMPLE_QUESTIONS = [
    "Who is the chairperson of Computer Science?",
    "What programs does Electrical Engineering offer?",
    "What are the admission requirements for Software Engineering?",
    "What are the lab facilities in Mechanical Engineering?",
    "What is the fee structure for BS programs?",
    "Tell me about the Civil Engineering department",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UET AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.user-bubble {
    background: #e3f2fd;
    border-left: 4px solid #1565c0;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
.bot-bubble {
    background: #f8f9fa;
    border-left: 4px solid #2e7d32;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
.out-of-scope {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    padding: 12px 16px;
    border-radius: 8px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "session_id"   not in st.session_state: st.session_state.session_id   = str(uuid.uuid4())
if "query_count"  not in st.session_state: st.session_state.query_count  = 0
if "total_time"   not in st.session_state: st.session_state.total_time   = 0.0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#1a237e,#1565c0);padding:20px 30px;
border-radius:12px;color:white;margin-bottom:20px;text-align:center;">
<h1 style="margin:0;font-size:2rem;">🎓 UET AI Assistant</h1>
<p style="margin:6px 0 0;opacity:0.85;">Accurate answers about UET departments · Powered by MCP + RAG</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    dept_filter = st.selectbox("Filter by Department (optional)", options=[
        "", "Computer Science", "Electrical Engineering", "Mechanical Engineering",
        "Civil Engineering", "Software Engineering", "Chemical Engineering",
        "Electronics", "Telecommunication", "Mathematics", "Physics",
    ])

    st.divider()
    st.header("📊 Session Stats")
    st.metric("Queries Asked", st.session_state.query_count)
    avg_time = (
        round(st.session_state.total_time / st.session_state.query_count, 2)
        if st.session_state.query_count else 0
    )
    st.metric("Avg Response Time", f"{avg_time}s")

    st.divider()
    st.header("💡 Sample Questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state["pending_query"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.query_count = 0
        st.session_state.total_time  = 0.0
        st.rerun()

    st.divider()
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        if r.status_code == 200 and r.json().get("agent_ready"):
            st.success("🟢 API Connected")
        else:
            st.warning("🟡 API Starting …")
    except Exception:
        st.error("🔴 API Unreachable\nStart: `python backend/main.py`")

# ── Chat display ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        # User bubble
        st.markdown(
            f'<div class="user-bubble">👤 <b>You:</b> {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        data      = msg["data"]
        is_valid  = data.get("is_valid", True)
        css_class = "bot-bubble" if is_valid else "out-of-scope"
        icon      = "🤖" if is_valid else "⚠️"
        time_s    = data.get("processing_time_s", 0)

        # Answer bubble — plain text only, no HTML tags leaking in
        st.markdown(
            f'<div class="{css_class}">'
            f'{icon} <b>UET Assistant</b> <small>({time_s}s)</small>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Answer text rendered by Streamlit (handles markdown properly)
        st.markdown(msg["content"])

        # Small meta line using native Streamlit caption
        num_chunks = data.get("num_chunks", 0)
        llm_time   = data.get("llm_time_s", 0)
        tools      = " · ".join(data.get("tools_used", []))
        sources    = " · ".join(data.get("sources", []))
        caption_parts = []
        if tools:
            caption_parts.append(f"🔧 {tools}")
        if num_chunks:
            caption_parts.append(f"📄 {num_chunks} chunks")
        if llm_time:
            caption_parts.append(f"⏱ LLM {llm_time}s")
        if sources:
            caption_parts.append(f"📚 {sources}")
        if caption_parts:
            st.caption(" | ".join(caption_parts))

        st.divider()

# ── Input ─────────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask anything about UET departments …") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("🔍 Searching and generating answer …"):
        try:
            payload = {
                "message":    user_input,
                "session_id": st.session_state.session_id,
                "department": dept_filter,
            }
            resp = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
            resp.raise_for_status()
            data   = resp.json()
            answer = data.get("answer", "No answer returned.")

            st.session_state.messages.append({
                "role":    "assistant",
                "content": answer,
                "data":    data,
            })
            st.session_state.query_count += 1
            st.session_state.total_time  += data.get("processing_time_s", 0)

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Run `python backend/main.py` first.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    st.rerun()