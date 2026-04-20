import streamlit as st
import os
import numpy as np
from openai import OpenAI
from supabase import create_client
from rank_bm25 import BM25Okapi

# --- Load keys from .streamlit/secrets.toml ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Page config ---
st.set_page_config(
    page_title="PCOS Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp { font-family: 'DM Sans', sans-serif; }

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.2);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(99,102,241,0.3);
    }

    .stats-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
    .stat-card {
        flex: 1;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .stat-number { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
    .stat-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 2px;
    }

    .answer-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .answer-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }
    .answer-text { font-size: 0.95rem; color: #334155; line-height: 1.75; }

    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.8rem;
        color: #475569;
        margin: 4px;
        transition: all 0.2s;
    }
    .source-chip:hover { background: #e2e8f0; border-color: #6366f1; color: #6366f1; }
    .source-dot { width: 6px; height: 6px; background: #22c55e; border-radius: 50%; }

    .context-card {
        background: #fafbfc;
        border-left: 3px solid #6366f1;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
        border-radius: 0 10px 10px 0;
        font-size: 0.85rem;
        color: #475569;
        line-height: 1.6;
    }
    .context-meta {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }

    .disclaimer-bar {
        background: #fffbeb;
        border: 1px solid #fde68a;
        padding: 0.75rem 1.25rem;
        border-radius: 10px;
        font-size: 0.8rem;
        color: #92400e;
        margin-top: 1.5rem;
    }

    .sidebar-section {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #6366f1;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.75rem;
    }

    .stTextInput > div > div > input {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.75rem 1rem !important;
        font-size: 0.95rem !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stButton > button {
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        padding: 0.4rem 0.75rem !important;
        border: 1px solid #e2e8f0 !important;
        background: #ffffff !important;
        color: #475569 !important;
    }
    .stButton > button:hover {
        background: #f1f5f9 !important;
        border-color: #6366f1 !important;
        color: #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# FUNCTIONS
# =============================================

def get_embedding(text):
    response = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


def vector_search(query, top_k=20):
    emb = get_embedding(query)
    result = supabase.rpc("match_documents", {
        "query_embedding": emb,
        "match_count": top_k,
        "filter": {}
    }).execute()
    return result.data


@st.cache_resource
def load_chunks_from_supabase():
    all_chunks = []
    offset = 0
    batch = 1000
    while True:
        result = (
            supabase.table("documents")
            .select("content, metadata")
            .range(offset, offset + batch - 1)
            .execute()
        )
        if not result.data:
            break
        all_chunks.extend([
            {"content": r["content"], "metadata": r["metadata"]}
            for r in result.data
        ])
        offset += batch
        if len(result.data) < batch:
            break
    return all_chunks


@st.cache_resource
def build_bm25_index(_chunks):
    corpus = [c["content"] for c in _chunks]
    tokenized = [doc.lower().split() for doc in corpus]
    return BM25Okapi(tokenized), corpus


def hybrid_search(query, chunks, bm25, corpus, top_k=5):
    vec_results = vector_search(query, top_k=20)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25_idx = np.argsort(bm25_scores)[-20:][::-1]

    scores = {}
    for rank, r in enumerate(vec_results):
        key = r["content"][:100]
        scores[key] = {"score": 1 / (rank + 60), "data": r}

    for rank, idx in enumerate(top_bm25_idx):
        key = corpus[idx][:100]
        if key in scores:
            scores[key]["score"] += 1 / (rank + 60)
        else:
            scores[key] = {
                "score": 1 / (rank + 60),
                "data": {
                    "content": corpus[idx],
                    "metadata": chunks[idx]["metadata"],
                    "similarity": 0
                }
            }

    ranked = sorted(scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [r["data"] for r in ranked]


def ask(question, chunks, bm25, corpus, top_k=5):
    results = hybrid_search(question, chunks, bm25, corpus, top_k=top_k)

    if not results:
        return {"answer": "No relevant documents found.", "sources": []}

    context = "\n\n".join([
        f"[{r['metadata']['pmcid']}] ({r['metadata']['section']}): {r['content']}"
        for r in results
    ])

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """You are a medical research assistant specializing in PCOS (Polycystic Ovary Syndrome).
Answer using ONLY the provided context. Cite sources using [PMC ID].
If context is insufficient, say so. Never give medical advice.
Distinguish between established findings and preliminary results.
Note sample sizes and study limitations when relevant."""},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.1
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [{
            "pmcid": r["metadata"]["pmcid"],
            "section": r["metadata"]["section"],
            "url": r["metadata"]["source"],
            "excerpt": r["content"][:200]
        } for r in results]
    }


# =============================================
# LOAD DATA
# =============================================

with st.spinner("Loading research papers from database..."):
    chunks = load_chunks_from_supabase()

if not chunks:
    st.error("No data found in Supabase. Run the ingestion pipeline first.")
    st.stop()

bm25, corpus = build_bm25_index(chunks)
paper_count = len(set(c["metadata"]["pmcid"] for c in chunks))


# =============================================
# HERO + STATS
# =============================================

st.markdown(f"""
<div class="hero">
    <p class="hero-title">🔬 PCOS Research Assistant</p>
    <p class="hero-sub">
        Ask evidence-based questions about Polycystic Ovary Syndrome.
        Answers are generated exclusively from {paper_count} peer-reviewed PubMed Central papers
        using hybrid retrieval (semantic + keyword search).
    </p>
    <span class="hero-badge">✦ Hybrid RAG · GPT-4o-mini · pgvector</span>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-number">{paper_count}</div>
        <div class="stat-label">Research Papers</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">{len(chunks):,}</div>
        <div class="stat-label">Text Chunks</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">Hybrid</div>
        <div class="stat-label">Search Method</div>
    </div>
    <div class="stat-card">
        <div class="stat-number">GPT-4o</div>
        <div class="stat-label">LLM Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)


# =============================================
# SIDEBAR
# =============================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-title">About this tool</div>
        <p style="font-size: 0.85rem; color: #64748b; line-height: 1.5; margin: 0;">
            Searches published PCOS research from PubMed Central.
            Every answer is grounded in peer-reviewed papers with source citations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieval depth", min_value=3, max_value=10, value=5,
                       help="Number of text chunks to retrieve per query")
    show_context = st.checkbox("Show retrieved chunks", value=False)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">Suggested questions</div>', unsafe_allow_html=True)

    suggestions = [
        "What are the Rotterdam criteria for diagnosing PCOS?",
        "How does insulin resistance contribute to PCOS?",
        "What treatments help PCOS-related infertility?",
        "Does metformin help with PCOS and how?",
        "What dietary changes improve PCOS outcomes?",
        "What is the link between PCOS and mental health?",
        "How do elevated androgens affect women with PCOS?",
    ]
    for s in suggestions:
        if st.button(s, key=s, use_container_width=True):
            st.session_state.query = s


# =============================================
# MAIN SEARCH
# =============================================

query = st.text_input(
    "Ask a research question about PCOS:",
    value=st.session_state.get("query", ""),
    placeholder="e.g., What lifestyle interventions help manage insulin resistance in PCOS?"
)

if query:
    with st.spinner("Searching papers + generating answer..."):
        result = ask(query, chunks, bm25, corpus, top_k=top_k)

    # Answer card
    st.markdown(f"""
    <div class="answer-card">
        <div class="answer-label">Research Answer</div>
        <div class="answer-text">{result["answer"]}</div>
    </div>
    """, unsafe_allow_html=True)

    # Source chips
    st.markdown("**Sources cited**")
    sources_html = ""
    for src in result["sources"]:
        sources_html += f"""
        <a href="{src['url']}" target="_blank" style="text-decoration: none;">
            <span class="source-chip">
                <span class="source-dot"></span>
                {src['pmcid']} · {src['section']}
            </span>
        </a>
        """
    st.markdown(sources_html, unsafe_allow_html=True)

    # Expandable source details
    with st.expander("View source details"):
        for i, src in enumerate(result["sources"], 1):
            st.markdown(f"""
**{i}. {src['pmcid']}** — _{src['section']}_
> {src['excerpt']}...

[Open in PubMed Central →]({src['url']})

---
""")

    # Show raw context chunks
    if show_context:
        st.markdown("**Retrieved chunks**")
        context_results = hybrid_search(query, chunks, bm25, corpus, top_k=top_k)
        for i, r in enumerate(context_results, 1):
            st.markdown(f"""
            <div class="context-card">
                <div class="context-meta">Chunk {i} · {r['metadata']['pmcid']} · {r['metadata']['section']}</div>
                {r['content'][:400]}...
            </div>
            """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-bar">
        ⚠️ <strong>Disclaimer:</strong>&nbsp; This is a research tool, not medical advice.
        All answers come from published PubMed Central papers. Consult a healthcare professional for medical decisions.
    </div>
    """, unsafe_allow_html=True)
