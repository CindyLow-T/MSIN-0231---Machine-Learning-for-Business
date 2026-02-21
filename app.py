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
# - Centralised constants so you can tune speed/cost/quality
# - "Report" pipeline and "Chatbot" pipeline are separate on purpose
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
MAX_CHUNKS_TO_EMBED = 25                  # cost/speed cap for chunks
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
# - set_page_config must be called before most Streamlit UI
# - Custom CSS is ONLY for nicer report typography
# ============================================================

st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="centered",
)

# Small CSS injection to make headings + spacing nicer in the report area
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
# - Streamlit reruns the whole script often (button clicks, inputs, etc.)
# - session_state is how we keep the report/chat persistent across reruns
# ============================================================

if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""  # stored only for this session (not on disk)

if "api_key_input" not in st.session_state:
    st.session_state["api_key_input"] = ""   # transient textbox value

if "report_ready" not in st.session_state:
    st.session_state["report_ready"] = False # toggles "show report + chatbot" section

if "saved_industry" not in st.session_state:
    st.session_state["saved_industry"] = ""  # the industry that produced the current report

if "saved_report" not in st.session_state:
    st.session_state["saved_report"] = ""    # generated report text (markdown-ish)

if "saved_context" not in st.session_state:
    st.session_state["saved_context"] = ""   # evidence text used for report generation

if "saved_pages" not in st.session_state:
    st.session_state["saved_pages"] = []     # list of {"title": ..., "url": ...}

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []    # list of {"role": "user"/"assistant", "content": ...}

# Most recent chatbot retrieval (collapsible UI without nested expander)
if "last_answer_sources" not in st.session_state:
    st.session_state["last_answer_sources"] = []  # list of {"title":..., "url":...}
if "last_answer_evidence" not in st.session_state:
    st.session_state["last_answer_evidence"] = "" # text evidence chunks used for last answer


# ============================================================
# Sidebar: LLM selection + API key input
# - LLM selection is fixed to one option (coursework requirement)
# - API key is saved into session_state on_change to avoid "sticky" textbox
# ============================================================

def save_key_and_clear():
    """Save the pasted API key into session_state, then clear the input field."""
    pasted = (st.session_state.get("api_key_input") or "").strip()
    if pasted:
        st.session_state["OPENAI_API_KEY"] = pasted  # store key for this session only
        st.session_state["api_key_input"] = ""       # clear textbox UI


with st.sidebar:
    st.header("⚙️ Settings")
    llm_choice = st.selectbox("Select LLM", options=[MODEL_LLM], index=0)  # fixed dropdown

    st.markdown("---")
    st.subheader("🔐 OpenAI API Key")

    st.text_input(
        "Paste your key here (required)",
        type="password",                   # hide input
        placeholder="sk-...",
        key="api_key_input",               # binds to session_state
        on_change=save_key_and_clear       # commit key when user pastes
    )

    if st.session_state["OPENAI_API_KEY"]:
        st.success("API key saved for this session.")
        st.caption("Refreshing the page clears the session key.")
    else:
        st.warning("Paste your OpenAI API key to enable report generation.")


api_ready = bool((st.session_state.get("OPENAI_API_KEY") or "").strip())  # quick boolean for buttons
if not api_ready:
    st.warning("🔐 Please paste your OpenAI API key in the sidebar before using the assistant.")


# ============================================================
# Helper functions
# - Small utilities (stop the app, chunk text, embed, etc.)
# - Keeping them here keeps the main UI flow readable
# ============================================================

def safe_stop(message: str):
    """Show an error and stop the Streamlit run immediately."""
    st.error(message)
    st.stop()


def normalize_key(s: str) -> str:
    """
    Lowercase and remove non-alphanumeric characters.
    This makes 'kpop' match 'K-pop', 'e-commerce' match 'ecommerce', etc.
    """
    s = (s or "").lower()                       # unify casing
    return re.sub(r"[^a-z0-9]+", "", s)          # drop punctuation/spaces


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between vectors a and b (safe for zero norms)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)  # |a||b|
    if denom == 0:
        return 0.0                                # avoid divide-by-zero
    return float(np.dot(a, b) / denom)            # cos(theta)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """
    Split long text into overlapping chunks.
    Overlap reduces the chance we lose meaning at chunk boundaries.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)      # cap chunk to size
        chunks.append(text[start:end])
        if end == len(text):
            break                               # reached end of text
        start = max(0, end - overlap)           # slide window with overlap

    return chunks


def stable_hash(items: list[str]) -> str:
    """Stable cache key for a list of strings (used by cached_embeddings)."""
    return hashlib.md5("||".join(items).encode("utf-8")).hexdigest()


def embed_batched(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """
    Embed a list of texts in batches to reduce API overhead.
    Returns a list of numpy vectors.
    """
    vectors = []
    try:
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i:i + EMBED_BATCH_SIZE]                           # batch slice
            resp = client.embeddings.create(model=MODEL_EMBED, input=batch) # API call
            vectors.extend([np.array(e.embedding, dtype=np.float32) for e in resp.data])
        return vectors
    except Exception as e:
        safe_stop(f"Embedding failed: {e}")  # fail fast (this is a core dependency)


@st.cache_data(show_spinner=False)
def cached_embeddings(texts: list[str], cache_key: str):
    """
    Cache embeddings so repeat runs (same query/chunks) are faster and cheaper.
    - We do NOT include the API key in cache args.
    - We fetch the key from session_state at runtime.
    """
    _ = cache_key  # cache_key is only to control cache invalidation

    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()
    if not api_key_local:
        safe_stop("Missing API key. Paste your OpenAI API key in the sidebar first.")

    client = OpenAI(api_key=api_key_local)      # client created inside cache function
    vecs = embed_batched(client, texts)         # embed all texts
    return [v.tolist() for v in vecs]           # return JSON-serialisable vectors


# ============================================================
# Wikipedia page filtering + scoring
# - Goal: prefer industry-level overview pages, not single titles
# - Negative patterns try to exclude specific entertainment/media pages
# - Positive keywords boost "industry/market/sector" type pages
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
    """Return True if the page title looks like a specific non-industry page."""
    t = (title or "").strip()
    return any(re.search(pat, t, flags=re.IGNORECASE) for pat in NEGATIVE_TITLE_PATTERNS)


def core_query(user_input: str) -> str:
    """
    Remove common industry intent words so 'kpop industry' -> 'kpop'.
    This makes matching more robust when users add 'industry/market/sector'.
    """
    q = (user_input or "").lower()
    for kw in POSITIVE_TITLE_KEYWORDS:
        q = re.sub(rf"\b{re.escape(kw)}\b", " ", q, flags=re.IGNORECASE)  # remove intent word
    q = re.sub(r"\s+", " ", q).strip()  # normalise whitespace
    return q


def title_score(title: str, user_industry: str) -> float:
    """
    Heuristic score for whether a Wikipedia title looks industry-relevant.
    Key idea: be robust to hyphens/spacing:
    - 'kpop' should match 'K-pop'
    """
    t_raw = (title or "").strip()
    t = t_raw.lower()
    q_raw = (user_industry or "").strip()
    q = q_raw.lower()

    q_core = core_query(q)  # strip "industry/market/sector..." from user input

    score = 0.0

    # A) boost for industry-ish keywords in title
    for kw in POSITIVE_TITLE_KEYWORDS:
        if kw in t:
            score += 2.0

    # B) normalization match (kpop vs k-pop) using normalize_key
    if q_core:
        qk = normalize_key(q_core)
        if qk and qk in normalize_key(t_raw):
            score += 3.0

    # C) token overlap on core query (lighter fuzzy match)
    if q_core:
        q_tokens = [x for x in re.split(r"\W+", q_core) if x]    # tokens from user
        t_tokens = set([x for x in re.split(r"\W+", t) if x])    # tokens from title
        overlap = sum(1 for tok in q_tokens if tok in t_tokens)  # count overlaps
        score += min(2.0, overlap * 0.7)                         # cap contribution

    # D) downweight list pages (often too granular)
    if t.startswith("list of"):
        score -= 2.0

    return score


def filter_industry_pages(docs, user_industry: str, k: int):
    """
    Report pipeline filter:
    - Remove titles that look like single items (negative patterns)
    - Score remaining pages, take top-k
    Returns:
      selected_docs, excluded_info
    """
    kept, excluded = [], []

    for d in docs:
        title = d.metadata.get("title", "") or ""     # Wikipedia title
        source = d.metadata.get("source", "") or ""   # Wikipedia URL

        if is_negative_title(title):
            excluded.append({"title": title, "source": source, "reason": "Excluded (likely non-industry page)"})
            continue

        kept.append(d)

    scored = [(title_score(d.metadata.get("title", "") or "", user_industry), d) for d in kept]
    scored.sort(key=lambda x: x[0], reverse=True)     # highest score first

    selected = [d for _, d in scored[:k]]             # take top-k after scoring
    return selected, excluded


# ============================================================
# Chatbot-only filtering (light filter; does not change report pipeline)
# - Chat should be more permissive: lists/companies can be useful evidence
# - We only exclude Wikipedia meta pages and disambiguations
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
    Do NOT exclude (company), List of, etc. (those can be relevant for Q&A).
    """
    kept = []
    for d in raw_docs:
        title = (d.metadata.get("title", "") or "").strip()
        if any(re.search(p, title, flags=re.IGNORECASE) for p in CHAT_EXCLUDE_TITLE_PATTERNS):
            continue
        kept.append(d)
    return kept[:min(k, len(kept))]  # safe slicing even if fewer docs exist


def likely_unrelated_to_industry(industry_term: str, sources: list[dict], evidence_text: str) -> bool:
    """
    Soft, evidence-based gate (for chatbot only):
    - If we have evidence, treat it as related (don’t over-block).
    - If evidence is empty AND none of the top source titles contains the industry's core term,
      then it’s likely unrelated (or Wikipedia can't retrieve relevant pages).
    This reduces false 'unrelated' claims.
    """
    if evidence_text and evidence_text.strip():
        return False  # if we have evidence chunks, assume it's relevant enough

    ind_core = normalize_key(core_query(industry_term))
    if not ind_core:
        return False  # no meaningful industry core term => don't block

    top_titles = [s.get("title", "") for s in (sources or [])[:12]]  # just inspect top few
    hits = 0
    for t in top_titles:
        if ind_core in normalize_key(t):
            hits += 1

    return hits == 0  # no title matches + no evidence => likely unrelated


# ============================================================
# Industry-intent gate - used only for the report generation entry
# - This is the "anti-random-name" protection
# - Uses quick Wikipedia retrieval + heuristic title scoring
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
    raw_docs = retriever.invoke(user_input)  # quick Wikipedia lookup

    if not raw_docs:
        return False, "", 0.0, [], []

    titles = [(d.metadata.get("title", "") or "").strip() for d in raw_docs]  # extract titles
    scored = [(title_score(t, user_input), t) for t in titles]
    scored.sort(key=lambda x: x[0], reverse=True)  # best score first

    best_score, best_title = scored[0]
    top_titles = [t for _, t in scored[:min(5, len(scored))]]  # show these on failure

    ok = best_score >= INTENT_MIN_SCORE  # threshold-based accept

    # Extra rule: if user explicitly said "industry/market/sector", be slightly more forgiving
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
        ok = True  # override: looks like a legit industry query but scored slightly low

    return ok, best_title, float(best_score), top_titles, raw_docs


# ============================================================
# Typo suggestion
# - Optional UX: if Wikipedia's top result is very similar, suggest it
# ============================================================

def suggest_industry_correction(user_input: str) -> str | None:
    retriever = WikipediaRetriever(top_k_results=3)
    docs = retriever.invoke(user_input)
    if not docs:
        return None

    top_title = (docs[0].metadata.get("title", "") or "").lower()
    similarity = difflib.SequenceMatcher(None, user_input.lower(), top_title).ratio()

    if similarity >= 0.75 and user_input.lower() != top_title:
        return top_title  # suggest Wikipedia's most likely intended title
    return None


# ============================================================
# Evidence retrieval (embeddings-based chunk selection)
# - Turns Wikipedia pages into chunks
# - Embeds query and chunks
# - Selects top-k chunks by cosine similarity (basic RAG retrieval step)
# ============================================================

def retrieve_relevant_context(docs, query: str, top_k_chunks: int):
    chunks, metas = [], []

    # Build the chunk pool across all pages
    for d in docs:
        title = d.metadata.get("title", "Wikipedia")  # fallback label
        source = d.metadata.get("source", "")         # URL for reference

        for ch in chunk_text(d.page_content, CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append(ch)
            metas.append({
                "title": title,
                "source": source,
                "preview": ch[:200].replace("\n", " ") + "..."  # short preview (debug/inspection)
            })

    if not chunks:
        return "", []

    # Hard cap to control embedding cost/time
    chunks = chunks[:MAX_CHUNKS_TO_EMBED]
    metas = metas[:MAX_CHUNKS_TO_EMBED]

    # Embed the query (1 vector)
    query_vec = np.array(
        cached_embeddings([query], stable_hash([query]))[0],
        dtype=np.float32
    )

    # Embed all chunks (N vectors)
    chunk_vecs = [
        np.array(v, dtype=np.float32)
        for v in cached_embeddings(chunks, stable_hash(chunks))
    ]

    # Score each chunk via cosine similarity vs query
    scores = [cosine_similarity(query_vec, v) for v in chunk_vecs]
    top_idx = np.argsort(scores)[::-1][:top_k_chunks]  # indices of best chunks

    # Format selected evidence chunks (and keep meta if needed)
    evidence_text, evidence_meta = [], []
    for i, idx in enumerate(top_idx, start=1):
        evidence_text.append(f"[Evidence {i}] {chunks[idx]}")      # actual context for LLM
        evidence_meta.append({**metas[idx], "score": float(scores[idx])})  # diagnostics

    return "\n\n".join(evidence_text), evidence_meta


def retrieve_chat_evidence(user_question: str, industry_term: str):
    """
    Chatbot only (Wikipedia-only RAG):
    - Retrieves fresh Wikipedia pages for the user's question
    - Uses a LIGHT filter (only removes meta/disambiguation)
    - Embedding-reranks chunks using an industry-aware rerank query
    Returns:
      evidence_text, sources(list of dict title/url)
    """
    retriever = WikipediaRetriever(top_k_results=CHAT_RETRIEVER_TOP_K)
    raw_docs = retriever.invoke(user_question)  # retrieve for the question, not the industry label

    if not raw_docs:
        return "", []

    # Light filtering for chat: keep things broad
    chat_docs = filter_chat_pages(raw_docs, k=CHAT_TOP_K_PAGES)
    if not chat_docs:
        chat_docs = raw_docs[:min(CHAT_TOP_K_PAGES, len(raw_docs))]  # fallback

    # Make rerank query industry-aware (helps pull industry-relevant chunks)
    rerank_query = f"{industry_term} industry: {user_question}"

    evidence_text, _ = retrieve_relevant_context(
        chat_docs,
        rerank_query,
        top_k_chunks=CHAT_TOP_K_EVIDENCE_CHUNKS
    )

    # Keep sources so user can inspect Wikipedia pages used
    sources = [
        {"title": d.metadata.get("title", "Wikipedia"), "url": d.metadata.get("source", "")}
        for d in chat_docs
    ]

    return evidence_text, sources


# ============================================================
# LLM report generation
# - Strict grounding: "use ONLY the evidence"
# - Continuation logic if the model hits max_tokens (finish_reason == "length")
# ============================================================

def generate_report(client: OpenAI, industry: str, evidence: str, page_titles: list[str]) -> str:
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

    # User prompt contains the industry + list of sources + the evidence text
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
        model=llm_choice,                   # fixed choice from sidebar
        temperature=TEMPERATURE,            # report tone/variation
        max_tokens=MAX_OUTPUT_TOKENS,       # cap output (tries to stay under 500 words)
        messages=messages,
    )

    text = resp.choices[0].message.content or ""       # main report text
    finish_reason = resp.choices[0].finish_reason      # why it stopped (length, stop, etc.)

    # If truncated, ask it to continue (without restarting)
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
            model=llm_choice,
            temperature=TEMPERATURE,
            max_tokens=300,                  # small continuation budget
            messages=messages,
        )

        text2 = resp2.choices[0].message.content or ""
        text = text + text2                 # append continuation

    return text


# ============================================================
# PDF export
# - Takes report text (markdown-ish) and turns it into a clean PDF
# - Uses ReportLab Platypus for layout flow (Paragraph/Spacer/ListFlowable)
# ============================================================

def build_pdf_bytes(title: str, report_markdown: str) -> bytes:
    buffer = io.BytesIO()  # in-memory output buffer

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()

    # Title / heading styles (tuned for readability)
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

    # Body paragraph style
    body_style = ParagraphStyle(
        name="BodyStyle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        spaceAfter=8,  # extra whitespace between paragraphs
    )

    # Bullet style (indent so bullets look clean)
    bullet_style = ParagraphStyle(
        name="BulletStyle",
        parent=styles["BodyText"],
        fontSize=11,
        leading=16,
        leftIndent=14,
    )

    # Build a "story" list for Platypus
    elements = []
    elements.append(Paragraph(title, title_style))
    elements.append(Spacer(1, 8))  # a little breathing room after title

    lines = (report_markdown or "").splitlines()  # parse report line-by-line

    for line in lines:
        line = line.strip()

        if not line:
            elements.append(Spacer(1, 6))  # blank lines => vertical spacing
            continue

        # Headings
        if line.startswith("### "):
            elements.append(Paragraph(line[4:], h3_style))
            continue
        if line.startswith("## "):
            elements.append(Paragraph(line[3:], h2_style))
            continue
        if line.startswith("# "):
            elements.append(Paragraph(line[2:], title_style))
            continue

        # Bullet lines (single bullet per line)
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

        # Convert **bold** markdown to ReportLab <b> tags
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        elements.append(Paragraph(line, body_style))

    doc.build(elements)            # render all elements into the PDF

    pdf = buffer.getvalue()        # bytes output
    buffer.close()
    return pdf


# ============================================================
# Main UI: industry input + run button
# - Using a form prevents reruns on each keystroke
# - submitted=True only when user clicks "Generate report"
# ============================================================

with st.form("industry_form"):
    industry_input = st.text_input(
        "Industry",
        placeholder="e.g. video game industry / kpop industry"
    )
    submitted = st.form_submit_button(
        "Generate report",
        disabled=not api_ready  # can't run without API key
    )


# ============================================================
# Generate report (only when button is clicked)
# - Industry-intent gate prevents random names from generating a report
# - If gate passes, run: retrieve -> filter -> embed/rerank -> LLM report
# ============================================================

if submitted:
    industry = (industry_input or "").strip()  # clean user input
    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()

    # Guardrails: key + non-empty input
    if not api_key_local:
        safe_stop("Missing API key. Please paste your OpenAI API key in the sidebar first.")
    if not industry:
        st.warning("Please enter an industry.")
        st.stop()

    # --- Industry intent gate BEFORE doing the full pipeline ---
    ok, best_title, best_score, top_titles, _ = industry_intent_gate(industry)
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
        st.stop()

    client = OpenAI(api_key=api_key_local)  # create OpenAI client for this run

    # Optional spelling correction suggestion
    suggestion = suggest_industry_correction(industry)
    if suggestion:
        st.info(f"Did you mean **{suggestion}**?")
        if st.button("Use suggested spelling"):
            industry = suggestion  # update industry term if user accepts

    retrieval_query = f"Provide an industry overview of the {industry}."  # query used for chunk ranking

    # --- Step 1: retrieve a broader set of candidate Wikipedia pages ---
    with st.status("Retrieving Wikipedia pages…", expanded=False):
        retriever = WikipediaRetriever(top_k_results=RETRIEVER_TOP_K)
        raw_docs = retriever.invoke(industry)

    if not raw_docs:
        safe_stop("No Wikipedia pages found for that term. Try a broader or more standard industry name.")

    # --- Step 2: filter + score pages to get top industry-level pages ---
    docs, excluded = filter_industry_pages(raw_docs, industry, k=TOP_K_PAGES)

    if not docs:
        safe_stop(
            "Wikipedia returned results, but none looked industry-level after filtering. "
            "Try a broader label like 'video game industry' or 'semiconductor industry'."
        )

    # --- Step 3: retrieve top evidence chunks + generate report ---
    with st.status("Generating industry report…", expanded=False):
        context, evidence_meta = retrieve_relevant_context(
            docs,
            retrieval_query,
            top_k_chunks=TOP_K_EVIDENCE_CHUNKS
        )
        page_titles = [d.metadata.get("title", "Wikipedia") for d in docs]  # list titles for prompt
        report = generate_report(client, industry, context, page_titles)

    # Persist results across reruns (so download/chat doesn't wipe the page)
    st.session_state["saved_industry"] = industry
    st.session_state["saved_report"] = report
    st.session_state["saved_context"] = context
    st.session_state["saved_pages"] = [
        {"title": d.metadata.get("title", ""), "url": d.metadata.get("source", "")}
        for d in docs
    ]
    st.session_state["chat_history"] = []              # reset chat for a fresh industry
    st.session_state["last_answer_sources"] = []       # reset last-answer evidence
    st.session_state["last_answer_evidence"] = ""
    st.session_state["report_ready"] = True            # unlock report + chatbot UI


# ============================================================
# Render report + PDF + chatbot whenever report_ready is True
# - This section runs on rerun too (so output persists)
# ============================================================

if st.session_state.get("report_ready"):
    industry = st.session_state.get("saved_industry", "")
    report = st.session_state.get("saved_report", "")
    pages = st.session_state.get("saved_pages", [])

    # Inform user if fewer than 5 pages survived filtering
    if len(pages) < TOP_K_PAGES:
        st.warning(
            f"Only {len(pages)} industry-level Wikipedia page(s) were found after filtering. "
            "The report may be less comprehensive."
        )

    # Show the final report's Wikipedia sources
    st.subheader("Top 5 relevant Wikipedia pages")
    for i, p in enumerate(pages, start=1):
        st.write(f"{i}. {p['title']}: {p['url']}")

    # Render report HTML wrapper (uses CSS from earlier)
    st.markdown("## Industry Report (500 words)")
    st.markdown(
        f"""
        <div class="report-container">
        {report}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build PDF bytes on the fly from the saved report
    pdf_title = f"{industry.title()} Industry Report"
    pdf_bytes = build_pdf_bytes(pdf_title, report)

    # Download button (Streamlit will rerun script on click, but session_state preserves UI)
    st.download_button(
        label="⬇️ Download as PDF",
        data=pdf_bytes,
        file_name=f"{industry.lower().replace(' ', '_')}_industry_report.pdf",
        mime="application/pdf",
    )

    # ============================================================
    # Chatbot: Wikipedia-only, dynamic retrieval per question
    # - No nested expanders (uses checkbox)
    # - Softer "unrelated" logic (only when truly likely unrelated)
    # ============================================================

    st.markdown("## Follow-up Q&A")

    api_key_local = (st.session_state.get("OPENAI_API_KEY") or "").strip()
    if not api_key_local:
        safe_stop("Missing API key. Please paste your OpenAI API key in the sidebar first.")
    client = OpenAI(api_key=api_key_local)  # separate client is fine; same key

    # Expander to keep Q&A area tidy under the report
    with st.expander("Ask questions about this industry", expanded=True):
        st.caption(
            "Each question retrieves additional Wikipedia pages relevant to your question. "
            "Answers are grounded ONLY in Wikipedia evidence retrieved for that question."
        )

        # Collapsible evidence + sources for the LAST answer (NO nested expander)
        if st.session_state["last_answer_sources"] or st.session_state["last_answer_evidence"]:
            show_last = st.checkbox(
                "Show evidence + Wikipedia pages used for the last answer",
                value=False
            )

            if show_last:
                # List Wikipedia pages used for the last answer
                if st.session_state["last_answer_sources"]:
                    st.write("**Wikipedia pages (last answer):**")
                    for s in st.session_state["last_answer_sources"]:
                        title = s.get("title", "Wikipedia")
                        url = s.get("url", "")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.write(f"- {title}")

                # Show the raw evidence chunks (debug/traceability)
                if st.session_state["last_answer_evidence"]:
                    st.write("**Evidence chunks (last answer):**")
                    st.text(st.session_state["last_answer_evidence"])

        # Render chat history (so the conversation stays visible)
        for m in st.session_state["chat_history"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        # Chat input (only returns a string when user submits)
        user_q = st.chat_input("Ask a follow-up question about the industry…")

        if user_q:
            # Save user message to history first (so it appears immediately on rerun)
            st.session_state["chat_history"].append({"role": "user", "content": user_q})

            # Dynamic Wikipedia retrieval for THIS question (industry-aware rerank)
            chat_evidence, chat_sources = retrieve_chat_evidence(user_q, industry)

            # Store for the collapsible panel (last answer)
            st.session_state["last_answer_sources"] = chat_sources
            st.session_state["last_answer_evidence"] = chat_evidence

            # Only warn as "unrelated" if it is very likely truly off-industry
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

            # Build a small sources list for the QA prompt
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

            # Keep only recent turns to avoid prompt bloat (cheap context window control)
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

            # Ask the model for the answer (strictly grounded by prompt rules)
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

            st.rerun()  # rerun to render the new message cleanly

        # Two convenience buttons below the chat area
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Q&A chat"):
                st.session_state["chat_history"] = []          # wipe conversation
                st.session_state["last_answer_sources"] = []   # wipe evidence panel
                st.session_state["last_answer_evidence"] = ""
                st.rerun()
        with col2:
            if st.button("Clear report + start over"):
                # Reset everything back to "fresh start" state
                st.session_state["report_ready"] = False
                st.session_state["saved_industry"] = ""
                st.session_state["saved_report"] = ""
                st.session_state["saved_context"] = ""
                st.session_state["saved_pages"] = []
                st.session_state["chat_history"] = []
                st.session_state["last_answer_sources"] = []
                st.session_state["last_answer_evidence"] = ""
                st.rerun()