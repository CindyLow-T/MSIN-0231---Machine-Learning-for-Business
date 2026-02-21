# ============================================================
# Market Research Assistant (Streamlit)
#
# What this app does :
# 1) Ask for an industry term (e.g., "video game industry")
# 2) Pull related Wikipedia pages
# 3) Industry-intent gate: reject non-industry queries (robust, handles "kpop industry")
# 4) Filter out pages that look too specific (e.g., a single game/movie)
# 5) Use embeddings to pick the most relevant evidence chunks
# 6) Ask an LLM to write a concise industry report (grounded in the evidence)
# 7) Offer a "Download as PDF" button for the generated report
# 8) Offer a Wikipedia-grounded chatbot for follow-up Q&A
#    - Chatbot dynamically retrieves additional Wikipedia pages per question (Wikipedia-only)
#    - Evidence + URLs are collapsible (NO nested expanders)
# ============================================================

# ----------------------------
# Imports (standard libs)
# ----------------------------
import re          # regex cleaning / matching
import io          # in-memory bytes buffer (PDF)
import hashlib     # stable cache keys
import difflib     # spelling similarity suggestions

# ----------------------------
# Imports (third-party)
# ----------------------------
import numpy as np             # vector math for embeddings + cosine similarity
import streamlit as st         # UI framework

from openai import OpenAI      # OpenAI client (chat + embeddings)
from langchain_community.retrievers import WikipediaRetriever  # Wikipedia retrieval

# ----------------------------
# Imports (PDF building via ReportLab Platypus)
# ----------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch


# ============================================================
# Configuration (models, retrieval sizes, chunking, caching)
# ============================================================

MODEL_LLM = "gpt-4o-mini"                 # fixed (only one option per coursework requirement)
MODEL_EMBED = "text-embedding-3-small"    # embeddings model for similarity search

TEMPERATURE = 0.5                         # report creativity / variance
MAX_OUTPUT_TOKENS = 900                   # report token cap (tries to stay <500 words)

TOP_K_PAGES = 5                           # final "industry-level" pages for the report
RETRIEVER_TOP_K = 15                      # raw wikipedia results fetched for the report pipeline

TOP_K_EVIDENCE_CHUNKS = 6                 # evidence chunks used in REPORT generation
CHUNK_SIZE = 1200                         # chunk length for embedding
CHUNK_OVERLAP = 150                       # overlap to reduce boundary loss
MAX_CHUNKS_TO_EMBED = 40                  # cost/speed cap for chunks
EMBED_BATCH_SIZE = 64                     # embedding API batch size

# Chatbot settings (Wikipedia-only, dynamic retrieval per question)
QA_TEMPERATURE = 0.2                      # keep QA deterministic / factual
QA_MAX_TOKENS = 450                       # answer length cap

CHAT_RETRIEVER_TOP_K = 18                 # raw wikipedia results fetched for each question
CHAT_TOP_K_PAGES = 10                     # pages used for QA evidence building
CHAT_TOP_K_EVIDENCE_CHUNKS = 8            # evidence chunks for QA answer

# Industry intent gate settings (cheap/fast sanity check before full pipeline)
INTENT_CHECK_TOP_K = 6
INTENT_MIN_SCORE = 2.5   # lowered to handle inputs like "kpop industry" reliably


# ============================================================
# Streamlit page setup (layout + styling)
# ============================================================

st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="centered",
)

# ---- Simple CSS to make the report look less "raw markdown" ----
st.markdown(
    """
    <style>
    .report-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        line-height: 1.65;
    }
    .report-container h1 { font-size: 30px; font-weight: 700; margin-bottom: 1.2rem; }
    .report-container h2 { font-size: 22px; font-weight: 700; margin-top: 1.4rem; margin-bottom: 0.6rem; }
    .report-container h3 { font-size: 18px; font-weight: 600; margin-top: 1.1rem; margin-bottom: 0.5rem; }
    .report-container p { margin-bottom: 0.9rem; }
    .report-container ul { margin-top: 0.4rem; margin-bottom: 0.9rem; }
    .report-container li { margin-bottom: 0.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Market Research Assistant")
st.caption(
    "Enter an industry. The app retrieves Wikipedia pages (prioritising industry-level pages) "
    "and generates a short report grounded only in those sources. "
    "A Wikipedia-grounded Q&A chatbot is available after the report."
)


# ============================================================
# Session state defaults (for reruns)
# ============================================================

# API key state
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
if "api_key_input" not in st.session_state:
    st.session_state["api_key_input"] = ""

# Report state
if "report_ready" not in st.session_state:
    st.session_state["report_ready"] = False
if "saved_industry" not in st.session_state:
    st.session_state["saved_industry"] = ""
if "saved_report" not in st.session_state:
    st.session_state["saved_report"] = ""
if "saved_context" not in st.session_state:
    st.session_state["saved_context"] = ""
if "saved_pages" not in st.session_state:
    st.session_state["saved_pages"] = []  # list of {"title":..., "url":...}

# Chat state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_answer_sources" not in st.session_state:
    st.session_state["last_answer_sources"] = []
if "last_answer_evidence" not in st.session_state:
    st.session_state["last_answer_evidence"] = ""

# Run state machine (loading/cancel/restart)
if "run_state" not in st.session_state:
    st.session_state["run_state"] = "idle"  # "idle" | "running"
if "run_industry" not in st.session_state:
    st.session_state["run_industry"] = ""
if "pending_industry" not in st.session_state:
    st.session_state["pending_industry"] = ""
if "run_in_progress" not in st.session_state:
    st.session_state["run_in_progress"] = False
if "cancel_requested" not in st.session_state:
    st.session_state["cancel_requested"] = False
if "cancel_notice" not in st.session_state:
    st.session_state["cancel_notice"] = ""

# Persist textbox value across reruns
if "industry_input" not in st.session_state:
    st.session_state["industry_input"] = ""


# ============================================================
# Sidebar: API key input
# ============================================================

def save_key_and_clear():
    """Save pasted key into session state, then clear the input box."""
    pasted = (st.session_state.get("api_key_input") or "").strip()
    if pasted:
        st.session_state["OPENAI_API_KEY"] = pasted
        st.session_state["api_key_input"] = ""


with st.sidebar:
    st.header("⚙️ Settings")
    llm_choice = st.selectbox("Select LLM", options=[MODEL_LLM], index=0)  # fixed to one model

    st.markdown("---")
    st.subheader("🔐 OpenAI API Key")

    st.text_input(
        "Paste your key here (required)",
        type="password",
        placeholder="sk-...",
        key="api_key_input",
        on_change=save_key_and_clear
    )

    if (st.session_state.get("OPENAI_API_KEY") or "").strip():
        st.success("API key saved for this session.")
        st.caption("Refreshing the page clears the session key.")
    else:
        st.warning("Paste your OpenAI API key to enable report generation. Refreshing the page clears the session key.")


api_ready = bool((st.session_state.get("OPENAI_API_KEY") or "").strip())
if not api_ready:
    st.warning("🔐 Please paste your OpenAI API key in the sidebar before using the assistant.")


# ============================================================
# Helper functions (stop + cancel control)
# ============================================================

def safe_stop(message: str):
    """Stop execution with a red error message."""
    st.error(message)
    st.stop()


class CancelRun(Exception):
    """Internal exception to abort a run cleanly."""
    pass


def request_cancel(message: str = "Generation cancelled."):
    """
    Request cancel and rerun immediately so UI updates.
    - Textbox stays editable
    - Cancel notice shows once
    """
    st.session_state["cancel_requested"] = True
    st.session_state["cancel_notice"] = message
    st.rerun()


def check_cancel():
    """
    Checkpoint-based cancel.
    Note: cannot interrupt a single blocking network call mid-flight;
    it cancels at the next checkpoint after that call returns.
    """
    if st.session_state.get("cancel_requested", False):
        raise CancelRun()


# ============================================================
# Helper functions (text + math)
# ============================================================

def normalize_key(s: str) -> str:
    """
    Lowercase and remove non-alphanumeric characters.
    This makes 'kpop' match 'K-pop', 'e-commerce' match 'ecommerce', etc.
    """
    s = (s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for embedding vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split long text into overlapping chunks for embeddings-based retrieval."""
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)

    return chunks


def stable_hash(items: list[str]) -> str:
    """Deterministic cache key for a list of strings."""
    return hashlib.md5("||".join(items).encode("utf-8")).hexdigest()


# ============================================================
# Embeddings helpers (batched + cached)
# ============================================================

def embed_batched(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """Create embeddings in batches to reduce API overhead."""
    vectors = []
    try:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]
            resp = client.embeddings.create(model=MODEL_EMBED, input=batch)
            vectors.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
        return vectors
    except Exception as e:
        safe_stop(f"Embedding failed: {e}")


@st.cache_data(show_spinner=False)
def cached_embeddings(texts: list[str], cache_key: str):
    """
    Cache embeddings so repeat runs (same query/chunks) are faster and cheaper.
    - We do NOT include the API key in cache args.
    - We fetch the key from session_state at runtime.
    """
    _ = cache_key  # only to influence cache key

    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()
    if not api_key_local:
        safe_stop("Missing API key. Paste your OpenAI API key in the sidebar first.")

    client = OpenAI(api_key=api_key_local)
    vecs = embed_batched(client, texts)
    return [v.tolist() for v in vecs]


# ============================================================
# Wikipedia page filtering + scoring (report pipeline)
# ============================================================

NEGATIVE_TITLE_PATTERNS = [
    r"\(video game\)$",
    r"\(film\)$",
    r"\(novel\)$",
    r"\(album\)$",
    r"\(song\)$",
    r"\(TV series\)$",
    r"\(television series\)$",
    r"\(board game\)$",
    r"\(manga\)$",
    r"\(anime\)$",
    r"\(comics\)$",
    r"\(character\)$",
    r"\(company\)$",
    r"^List of ",
    r"^Outline of ",
]

POSITIVE_TITLE_KEYWORDS = [
    "industry", "market", "sector", "economy", "business", "trade", "manufacturing",
    "supply chain", "value chain", "retail", "commerce"
]


def is_negative_title(title: str) -> bool:
    """Exclude pages that are likely too specific / not industry-level."""
    t = (title or "").strip()
    return any(re.search(pat, t, flags=re.IGNORECASE) for pat in NEGATIVE_TITLE_PATTERNS)


def core_query(user_input: str) -> str:
    """Remove common industry intent words so 'kpop industry' -> 'kpop'."""
    q = (user_input or "").lower()
    for kw in POSITIVE_TITLE_KEYWORDS:
        q = re.sub(rf"\b{re.escape(kw)}\b", " ", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def title_score(title: str, user_industry: str) -> float:
    """
    Score how 'industry-level' a title looks, robust to hyphens/spacing.
    """
    t_raw = (title or "").strip()
    t = t_raw.lower()
    q_raw = (user_industry or "").strip()
    q = q_raw.lower()

    q_core = core_query(q)

    score = 0.0

    # A) boost for industry-ish keywords in title
    for kw in POSITIVE_TITLE_KEYWORDS:
        if kw in t:
            score += 2.0

    # B) normalization match (kpop vs k-pop)
    if q_core:
        qk = normalize_key(q_core)
        if qk and qk in normalize_key(t_raw):
            score += 3.0

    # C) token overlap on core query
    if q_core:
        q_tokens = [x for x in re.split(r"\W+", q_core) if x]
        t_tokens = set([x for x in re.split(r"\W+", t) if x])
        overlap = sum(1 for tok in q_tokens if tok in t_tokens)
        score += min(2.0, overlap * 0.7)

    # D) downweight list pages
    if t.startswith("list of"):
        score -= 2.0

    return score


def filter_industry_pages(docs, user_industry: str, k: int):
    """Filter out overly-specific pages, then rank remaining pages by relevance."""
    kept, excluded = [], []

    for d in docs:
        title = d.metadata.get("title", "") or ""
        source = d.metadata.get("source", "") or ""

        if is_negative_title(title):
            excluded.append({"title": title, "source": source, "reason": "Excluded (likely non-industry page)"})
            continue

        kept.append(d)

    scored = [(title_score(d.metadata.get("title", "") or "", user_industry), d) for d in kept]
    scored.sort(key=lambda x: x[0], reverse=True)

    selected = [d for _, d in scored[:k]]
    return selected, excluded


# ============================================================
# Chatbot-only filtering (light filter)
# ============================================================

CHAT_EXCLUDE_TITLE_PATTERNS = [
    r"\(disambiguation\)$",
    r"^Help:",
    r"^Template:",
    r"^Wikipedia:",
    r"^Portal:",
]


def filter_chat_pages(raw_docs, k: int):
    """
    Chatbot only: keep the net wide.
    Only remove Wikipedia meta/disambiguation pages.
    """
    kept = []
    for d in raw_docs:
        title = (d.metadata.get("title", "") or "").strip()
        if any(re.search(p, title, flags=re.IGNORECASE) for p in CHAT_EXCLUDE_TITLE_PATTERNS):
            continue
        kept.append(d)
    return kept[:min(k, len(kept))]


def likely_unrelated_to_industry(industry_term: str, sources: list[dict], evidence_text: str) -> bool:
    """
    Soft, evidence-based gate:
    - If we have evidence, treat it as related.
    - If evidence is empty AND none of the top source titles contains the industry's core term,
      then it's likely unrelated (or Wikipedia can't retrieve relevant pages).
    """
    if evidence_text and evidence_text.strip():
        return False

    ind_core = normalize_key(core_query(industry_term))
    if not ind_core:
        return False

    top_titles = [s.get("title", "") for s in (sources or [])[:12]]
    hits = 0
    for t in top_titles:
        if ind_core in normalize_key(t):
            hits += 1

    return hits == 0


# ============================================================
# Industry intent gate (report entry gate)
# ============================================================

def industry_intent_gate(user_input: str):
    """
    Robust intent gate:
    - Retrieve top Wikipedia results
    - Score titles for industry relevance
    - Accept if:
        (1) best score >= threshold OR
        (2) user typed 'industry/market/sector' AND top titles match the core term (normalized)
    """
    retriever = WikipediaRetriever(top_k_results=INTENT_CHECK_TOP_K)
    raw_docs = retriever.invoke(user_input)

    if not raw_docs:
        return False, "", 0.0, [], []

    titles = [(d.metadata.get("title", "") or "").strip() for d in raw_docs]
    scored = [(title_score(t, user_input), t) for t in titles]
    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_title = scored[0]
    top_titles = [t for _, t in scored[:min(5, len(scored))]]

    ok = best_score >= INTENT_MIN_SCORE

    q = (user_input or "").lower()
    user_has_intent_word = any(w in q for w in ["industry", "market", "sector"])
    q_core = core_query(user_input)
    qk = normalize_key(q_core)

    core_matches_top = False
    if qk:
        for t in top_titles:
            if qk in normalize_key(t):
                core_matches_top = True
                break

    if (not ok) and user_has_intent_word and core_matches_top:
        ok = True

    return ok, best_title, float(best_score), top_titles, raw_docs


# ============================================================
# Typo suggestion (optional UI hint)
# ============================================================

def suggest_industry_correction(user_input: str) -> str | None:
    """Suggest a closer Wikipedia title if user likely misspelled the term."""
    retriever = WikipediaRetriever(top_k_results=3)
    docs = retriever.invoke(user_input)
    if not docs:
        return None

    top_title = (docs[0].metadata.get("title", "") or "").lower()
    similarity = difflib.SequenceMatcher(None, user_input.lower(), top_title).ratio()

    if similarity >= 0.75 and user_input.lower() != top_title:
        return top_title
    return None


# ============================================================
# Evidence retrieval (embeddings-based chunk selection)
# ============================================================

def retrieve_relevant_context(docs, query: str, top_k_chunks: int):
    """Chunk documents, embed chunks, and return the most relevant evidence for the query."""
    chunks, metas = [], []

    for d in docs:
        title = d.metadata.get("title", "Wikipedia")
        source = d.metadata.get("source", "")

        for ch in chunk_text(d.page_content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(ch)
            metas.append({
                "title": title,
                "source": source,
                "preview": ch[:200].replace("\n", " ") + "..."
            })

    if not chunks:
        return "", []

    chunks = chunks[:MAX_CHUNKS_TO_EMBED]
    metas = metas[:MAX_CHUNKS_TO_EMBED]

    query_vec = np.array(
        cached_embeddings([query], stable_hash([query]))[0],
        dtype=np.float32
    )

    chunk_vecs = [
        np.array(v, dtype=np.float32)
        for v in cached_embeddings(chunks, stable_hash(chunks))
    ]

    scores = [cosine_similarity(query_vec, v) for v in chunk_vecs]
    top_idx = np.argsort(scores)[::-1][:top_k_chunks]

    evidence_text, evidence_meta = [], []
    for i, idx in enumerate(top_idx, start=1):
        evidence_text.append(f"[Evidence {i}] {chunks[idx]}")
        evidence_meta.append({**metas[idx], "score": float(scores[idx])})

    return "\n\n".join(evidence_text), evidence_meta


def retrieve_chat_evidence(user_question: str, industry_term: str):
    """
    Chatbot only (Wikipedia-only RAG):
    - Retrieves fresh Wikipedia pages for the user's question
    - Uses a LIGHT filter (only removes meta/disambiguation)
    - Embedding-reranks chunks using an industry-aware rerank query
    """
    retriever = WikipediaRetriever(top_k_results=CHAT_RETRIEVER_TOP_K)
    raw_docs = retriever.invoke(user_question)

    if not raw_docs:
        return "", []

    chat_docs = filter_chat_pages(raw_docs, k=CHAT_TOP_K_PAGES)
    if not chat_docs:
        chat_docs = raw_docs[:min(CHAT_TOP_K_PAGES, len(raw_docs))]

    rerank_query = f"{industry_term} industry: {user_question}"

    evidence_text, _ = retrieve_relevant_context(
        chat_docs,
        rerank_query,
        top_k_chunks=CHAT_TOP_K_EVIDENCE_CHUNKS
    )

    sources = [
        {"title": d.metadata.get("title", "Wikipedia"), "url": d.metadata.get("source", "")}
        for d in chat_docs
    ]

    return evidence_text, sources


# ============================================================
# LLM report generation
# ============================================================

def generate_report(client: OpenAI, industry: str, evidence: str, page_titles: list[str]) -> str:
    """Create the report using ONLY the evidence retrieved from Wikipedia."""
    sources_block = "\n".join([f"- {t}" for t in page_titles]) if page_titles else "- (none)"

    system_prompt = (
        "You are a market research assistant supporting a corporate business analyst.\n"
        "Rules:\n"
        "- The report must be fewer than 500 words.\n"
        "- Use ONLY the provided Wikipedia evidence.\n"
        "- Do NOT invent facts, statistics, or citations.\n"
        "- If evidence is insufficient, say so.\n"
        "- Use clear headings and bullet points.\n"
        "- If you don't know, say you don't know.\n"
    )

    user_prompt = f"""
Industry:
{industry}

Wikipedia sources (titles):
{sources_block}

Wikipedia evidence:
{evidence}

Task:
Write a concise industry report with:
1) Executive summary (5 bullets)
2) Industry definition & scope
3) Value chain (high level)
4) Market structure & key players
5) Demand drivers
6) Supply-side dynamics
7) Trends & disruptions
8) Regulation / risks
9) Strategic implications for a large corporation
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=MODEL_LLM,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=messages,
    )

    text = resp.choices[0].message.content or ""
    finish_reason = resp.choices[0].finish_reason

    # If truncated, ask the model to continue (still enforcing <500 words overall)
    if finish_reason == "length":
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                "Continue exactly where you left off. "
                "Do not restart. Finish the last incomplete sentence first. "
                "Keep the entire report under 500 words total."
            )
        })

        resp2 = client.chat.completions.create(
            model=MODEL_LLM,
            temperature=TEMPERATURE,
            max_tokens=300,
            messages=messages,
        )

        text2 = resp2.choices[0].message.content or ""
        text = text + text2

    return text


# ============================================================
# PDF export
# ============================================================

def build_pdf_bytes(title: str, report_markdown: str) -> bytes:
    """Render the report into a nicely formatted PDF."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name="TitleStyle",
        parent=styles["Heading1"],
        fontSize=20,
        spaceAfter=14,
        textColor=colors.black,
    )

    h2_style = ParagraphStyle(
        name="H2Style",
        parent=styles["Heading2"],
        fontSize=14,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.black,
    )

    h3_style = ParagraphStyle(
        name="H3Style",
        parent=styles["Heading3"],
        fontSize=12,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.black,
    )

    body_style = ParagraphStyle(
        name="BodyStyle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        spaceAfter=8,
    )

    bullet_style = ParagraphStyle(
        name="BulletStyle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        leftIndent=14,
    )

    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 8))

    lines = (report_markdown or "").splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            elements.append(Spacer(1, 6))
            continue

        if line.startswith("### "):
            elements.append(Paragraph(line[4:], h3_style))
            continue
        if line.startswith("## "):
            elements.append(Paragraph(line[3:], h2_style))
            continue
        if line.startswith("# "):
            elements.append(Paragraph(line[2:], title_style))
            continue

        if line.startswith("- ") or line.startswith("* "):
            bullet_text = line[2:].strip()
            elements.append(
                ListFlowable(
                    [ListItem(Paragraph(bullet_text, bullet_style))],
                    bulletType="bullet",
                    leftIndent=18,
                )
            )
            continue

        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        elements.append(Paragraph(line, body_style))

    doc.build(elements)

    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# ============================================================
# UI: cancel notice + cancel button + industry form
# ============================================================

# Show one-time cancel notice (if user cancelled)
if st.session_state.get("cancel_notice"):
    st.warning(st.session_state["cancel_notice"])
    st.session_state["cancel_notice"] = ""

# Show cancel button when running
if st.session_state.get("run_state") == "running":
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Cancel generation"):
            request_cancel("Generation cancelled.")
    with colB:
        st.info(
            "Loading… You can keep typing in the box. "
            "Only clicking Generate/Enter will submit a new run."
        )

# Main input form (textbox always editable)
with st.form("industry_form"):
    st.text_input(
        "Industry",
        placeholder="e.g. video game industry / kpop industry",
        key="industry_input",
        disabled=False
    )
    submitted = st.form_submit_button("Generate report", disabled=not api_ready)


# ============================================================
# Submit logic:
# - If currently running, queue the new industry + cancel current
# - Otherwise start a run and rerun so cancel button appears immediately
# ============================================================

if submitted:
    new_industry = (st.session_state.get("industry_input") or "").strip()

    if not new_industry:
        st.warning("Please enter an industry.")
        st.stop()

    # If already running, queue the new request and cancel old run
    if st.session_state.get("run_state") == "running":
        st.session_state["pending_industry"] = new_industry
        request_cancel("Cancelling current run and preparing the new industry…")

    # Start a new run
    st.session_state["run_state"] = "running"
    st.session_state["run_industry"] = new_industry
    st.session_state["cancel_requested"] = False

    # Rerun immediately so cancel button appears right away
    st.rerun()


# ============================================================
# Run pipeline (executes only when run_state == "running")
# ============================================================

if st.session_state.get("run_state") == "running" and not st.session_state.get("run_in_progress", False):
    st.session_state["run_in_progress"] = True

    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()
    industry = (st.session_state.get("run_industry") or "").strip()

    try:
        if not api_key_local:
            safe_stop("Missing API key. Please paste your OpenAI API key in the sidebar first.")

        with st.spinner("Loading… retrieving Wikipedia pages and generating the report."):
            check_cancel()  # checkpoint 0

            # Industry intent gate
            ok, best_title, best_score, top_titles, _ = industry_intent_gate(industry)
            check_cancel()  # checkpoint 1

            if not ok:
                st.warning(
                    "That input doesn't look like an industry term based on Wikipedia's top matches.\n\n"
                    "Try a broader industry label like:\n"
                    "- video game industry\n"
                    "- semiconductor industry\n"
                    "- airline industry\n"
                    "- fast fashion market\n\n"
                    "Top Wikipedia matches for your input were:\n"
                    + "\n".join([f"- {t}" for t in top_titles])
                )
                st.session_state["run_state"] = "idle"
                st.stop()

            client = OpenAI(api_key=api_key_local)
            check_cancel()  # checkpoint 2

            # Spelling suggestion (non-blocking hint)
            suggestion = suggest_industry_correction(industry)
            if suggestion:
                st.info(f"Spelling hint: Wikipedia top match is **{suggestion}** (you can re-submit if needed).")

            retrieval_query = f"Provide an industry overview of the {industry}."
            check_cancel()  # checkpoint 3

            # Retrieve pages
            with st.status("Retrieving Wikipedia pages…", expanded=False):
                retriever = WikipediaRetriever(top_k_results=RETRIEVER_TOP_K)
                raw_docs = retriever.invoke(industry)

            check_cancel()  # checkpoint 4

            if not raw_docs:
                safe_stop("No Wikipedia pages found for that term. Try a broader or more standard industry name.")

            # Filter to industry-level pages
            docs, excluded = filter_industry_pages(raw_docs, industry, k=TOP_K_PAGES)
            check_cancel()  # checkpoint 5

            if not docs:
                safe_stop(
                    "Wikipedia returned results, but none looked industry-level after filtering. "
                    "Try a broader label like 'video game industry' or 'semiconductor industry'."
                )

            # Evidence retrieval + report writing
            with st.status("Generating industry report…", expanded=False):
                context, evidence_meta = retrieve_relevant_context(
                    docs,
                    retrieval_query,
                    top_k_chunks=TOP_K_EVIDENCE_CHUNKS
                )
                check_cancel()  # checkpoint 6

                page_titles = [d.metadata.get("title", "Wikipedia") for d in docs]
                report = generate_report(client, industry, context, page_titles)
                check_cancel()  # checkpoint 7

            # Save results for rendering
            st.session_state["saved_industry"] = industry
            st.session_state["saved_report"] = report
            st.session_state["saved_context"] = context
            st.session_state["saved_pages"] = [
                {"title": d.metadata.get("title", ""), "url": d.metadata.get("source", "")}
                for d in docs
            ]

            # Reset chatbot state for new report
            st.session_state["chat_history"] = []
            st.session_state["last_answer_sources"] = []
            st.session_state["last_answer_evidence"] = ""
            st.session_state["report_ready"] = True

            # End run
            st.session_state["run_state"] = "idle"

    except CancelRun:
        # Cancelled: end run, unlock UI
        st.session_state["run_state"] = "idle"

    finally:
        st.session_state["run_in_progress"] = False
        st.session_state["cancel_requested"] = False

        # If user queued a new industry while running, start it automatically
        pending = (st.session_state.get("pending_industry") or "").strip()
        if pending:
            st.session_state["pending_industry"] = ""
            st.session_state["run_state"] = "running"
            st.session_state["run_industry"] = pending
            st.rerun()


# ============================================================
# Render report + PDF + chatbot whenever report_ready is True
# ============================================================

if st.session_state.get("report_ready", False):
    industry = st.session_state.get("saved_industry", "")
    report = st.session_state.get("saved_report", "")
    pages = st.session_state.get("saved_pages", [])

    if len(pages) < TOP_K_PAGES:
        st.warning(
            f"Only {len(pages)} industry-level Wikipedia page(s) were found after filtering. "
            "The report may be less comprehensive."
        )

    st.subheader("Top 5 relevant Wikipedia pages")
    for i, p in enumerate(pages, start=1):
        st.write(f"{i}. {p['title']}: {p['url']}")

    st.markdown("## Industry Report (500 words)")
    st.markdown(
        f"""
        <div class="report-container">
        {report}
        </div>
        """,
        unsafe_allow_html=True
    )

    # PDF download
    pdf_title = f"{industry.title()} Industry Report"
    pdf_bytes = build_pdf_bytes(pdf_title, report)

    st.download_button(
        label="⬇️ Download as PDF",
        data=pdf_bytes,
        file_name=f"{industry.lower().replace(' ', '_')}_industry_report.pdf",
        mime="application/pdf",
    )

    # ============================================================
    # Chatbot: Wikipedia-only, dynamic retrieval per question
    # ============================================================

    st.markdown("## Follow-up Q&A")

    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()
    if not api_key_local:
        safe_stop("Missing API key. Please paste your OpenAI API key in the sidebar first.")
    client = OpenAI(api_key=api_key_local)

    with st.expander("Ask questions about this industry", expanded=True):
        st.caption(
            "Each question retrieves additional Wikipedia pages relevant to your question. "
            "Answers are grounded only in Wikipedia evidence retrieved for that question."
        )

        # Collapsible evidence + sources for LAST answer (no nested expander)
        if st.session_state.get("last_answer_sources") or st.session_state.get("last_answer_evidence"):
            show_last = st.checkbox("Show evidence + Wikipedia pages used for the last answer", value=False)

            if show_last:
                if st.session_state.get("last_answer_sources"):
                    st.write("**Wikipedia pages (last answer):**")
                    for s in st.session_state["last_answer_sources"]:
                        title = s.get("title", "Wikipedia")
                        url = s.get("url", "")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.write(f"- {title}")

                if st.session_state.get("last_answer_evidence"):
                    st.write("**Evidence chunks (last answer):**")
                    st.text(st.session_state["last_answer_evidence"])

        # Show chat history
        for m in st.session_state.get("chat_history", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a follow-up question about the industry…")

        if user_q:
            st.session_state["chat_history"].append({"role": "user", "content": user_q})

            # Retrieve evidence + sources dynamically for this question
            chat_evidence, chat_sources = retrieve_chat_evidence(user_q, industry)

            # Store for collapsible panel
            st.session_state["last_answer_sources"] = chat_sources
            st.session_state["last_answer_evidence"] = chat_evidence

            # Soft unrelated check (only when truly likely unrelated)
            if likely_unrelated_to_industry(industry, chat_sources, chat_evidence):
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": (
                        "This question appears unrelated to the selected industry, "
                        "or Wikipedia retrieval did not return enough industry-relevant evidence. "
                        "Please rephrase the question to explicitly connect it to the industry's market, value chain, "
                        "key players, regulation, trends, or business model."
                    )
                })
                st.rerun()

            sources_block = "\n".join([f"- {s['title']}" for s in chat_sources]) if chat_sources else "- (none)"

            qa_system = (
                "You are a market research assistant.\n"
                "Rules:\n"
                "- Answer ONLY using the provided Wikipedia evidence.\n"
                "- Do NOT invent facts, numbers, or citations.\n"
                "- If evidence is limited, give the best possible answer using ONLY what is present and "
                "clearly state what cannot be concluded from Wikipedia evidence.\n"
                "- Be concise, structured, and business-relevant.\n"
            )

            recent_turns = st.session_state["chat_history"][-8:]
            convo_text = "\n".join([f'{x["role"].upper()}: {x["content"]}' for x in recent_turns])

            qa_user = f"""
Original industry (report): {industry}

Wikipedia sources (titles) retrieved for this question:
{sources_block}

Wikipedia evidence:
{chat_evidence}

Conversation so far:
{convo_text}

User question:
{user_q}

Task:
Answer the user question grounded strictly in the evidence above.
If relevant, quote short phrases from evidence (no long quotes).
"""

            qa_resp = client.chat.completions.create(
                model=MODEL_LLM,
                temperature=QA_TEMPERATURE,
                max_tokens=QA_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": qa_system},
                    {"role": "user", "content": qa_user},
                ],
            )

            answer = qa_resp.choices[0].message.content or "I don't know based on the provided Wikipedia evidence."
            st.session_state["chat_history"].append({"role": "assistant", "content": answer})

            st.rerun()

        # Clear buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Q&A chat"):
                st.session_state["chat_history"] = []
                st.session_state["last_answer_sources"] = []
                st.session_state["last_answer_evidence"] = ""
                st.rerun()
        with col2:
            if st.button("Clear report + start over"):
                st.session_state["report_ready"] = False
                st.session_state["saved_industry"] = ""
                st.session_state["saved_report"] = ""
                st.session_state["saved_context"] = ""
                st.session_state["saved_pages"] = []
                st.session_state["chat_history"] = []
                st.session_state["last_answer_sources"] = []
                st.session_state["last_answer_evidence"] = ""
                st.session_state["run_state"] = "idle"
                st.session_state["run_industry"] = ""
                st.session_state["pending_industry"] = ""
                st.session_state["cancel_requested"] = False
                st.session_state["cancel_notice"] = ""
                st.rerun()