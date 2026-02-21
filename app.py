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
# Session state defaults (so reruns keep the page content)
# ============================================================

# --- API key state ---
if "OPENAI_API_KEY" not in st.session_state:
    st.session_state["OPENAI_API_KEY"] = ""
if "api_key_input" not in st.session_state:
    st.session_state["api_key_input"] = ""

# --- Report outputs/state ---
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

# --- Chat state ---
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_answer_sources" not in st.session_state:
    st.session_state["last_answer_sources"] = []
if "last_answer_evidence" not in st.session_state:
    st.session_state["last_answer_evidence"] = ""

# --- Run state machine (loading/cancel/restart) ---
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

# --- Make textbox value persist across reruns ---
if "industry_input" not in st.session_state:
    st.session_state["industry_input"] = ""


# ============================================================
# Sidebar: LLM selection + API key input
# ============================================================

def save_key_and_clear():
    """Save pasted key into session state, then clear the input box."""
    pasted = (st.session_state.get("api_key_input") or "").strip()
    if pasted:
        st.session_state["OPENAI_API_KEY"] = pasted
        st.session_state["api_key_input"] = ""


with st.sidebar:
    st.header("⚙️ Settings")
    llm_choice = st.selectbox("Select LLM", options=[MODEL_LLM], index=0)

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
# Helper functions (control flow + small utilities)
# - Keep your existing helpers above if you already have them.
# - Only the cancel/stop logic is new / important here.
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
    Request cancel and rerun immediately so UI updates (textbox stays editable).
    """
    st.session_state["cancel_requested"] = True
    st.session_state["cancel_notice"] = message
    st.rerun()


def check_cancel():
    """
    Checkpoint-based cancel.
    - Can't interrupt a single blocking network call mid-flight.
    - Stops at the next checkpoint.
    """
    if st.session_state.get("cancel_requested", False):
        raise CancelRun()


# ============================================================
# Your existing functions should remain unchanged below:
# - normalize_key, cosine_similarity, chunk_text, stable_hash
# - embed_batched, cached_embeddings
# - scoring/filtering (title_score, filter_industry_pages, etc.)
# - industry_intent_gate, suggest_industry_correction
# - retrieve_relevant_context, retrieve_chat_evidence
# - generate_report, build_pdf_bytes
# ============================================================


# ============================================================
# UI: cancel notice + cancel button + industry form
# ============================================================

# One-time cancel notice (shows after a cancel rerun)
if st.session_state.get("cancel_notice"):
    st.warning(st.session_state["cancel_notice"])
    st.session_state["cancel_notice"] = ""

# Cancel button is always visible while "running"
if st.session_state.get("run_state") == "running":
    colA, colB = st.columns([1, 3])
    with colA:
        if st.button("Cancel generation"):
            # stop the current run ASAP (at next checkpoint)
            request_cancel("Generation cancelled.")
    with colB:
        st.info(
            "Loading… You can keep typing in the box. "
            "Only clicking Generate/Enter will submit a new run."
        )

# Form: textbox stays editable at all times
with st.form("industry_form"):
    st.text_input(
        "Industry",
        placeholder="e.g. video game industry / kpop industry",
        key="industry_input",
        disabled=False
    )
    submitted = st.form_submit_button(
        "Generate report",
        disabled=not api_ready
    )


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

    if st.session_state.get("run_state") == "running":
        # user submitted again mid-run -> queue the new industry and cancel current run
        st.session_state["pending_industry"] = new_industry
        request_cancel("Cancelling current run and preparing the new industry…")

    # start a new run
    st.session_state["run_state"] = "running"
    st.session_state["run_industry"] = new_industry
    st.session_state["cancel_requested"] = False

    # force immediate rerun so the cancel button + loading UI renders now
    st.rerun()


# ============================================================
# Run pipeline:
# - Executes only when run_state == "running"
# - Uses checkpoints so cancel works between steps
# - Auto-starts pending industry after cancel
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

            # --- Industry intent gate ---
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

            # Optional spelling suggestion (kept from your existing logic)
            suggestion = suggest_industry_correction(industry)
            if suggestion:
                st.info(f"Did you mean **{suggestion}**?")
                # NOTE: Button inside a running pipeline can cause reruns; keeping it as-is from your design.
                if st.button("Use suggested spelling"):
                    industry = suggestion
                    st.session_state["run_industry"] = industry

            retrieval_query = f"Provide an industry overview of the {industry}."
            check_cancel()  # checkpoint 3

            # --- Retrieve Wikipedia pages ---
            with st.status("Retrieving Wikipedia pages…", expanded=False):
                retriever = WikipediaRetriever(top_k_results=RETRIEVER_TOP_K)
                raw_docs = retriever.invoke(industry)

            check_cancel()  # checkpoint 4

            if not raw_docs:
                safe_stop("No Wikipedia pages found for that term. Try a broader or more standard industry name.")

            # --- Filter to industry-level pages ---
            docs, excluded = filter_industry_pages(raw_docs, industry, k=TOP_K_PAGES)
            check_cancel()  # checkpoint 5

            if not docs:
                safe_stop(
                    "Wikipedia returned results, but none looked industry-level after filtering. "
                    "Try a broader label like 'video game industry' or 'semiconductor industry'."
                )

            # --- Evidence + report generation ---
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

            # --- Save outputs for rendering ---
            st.session_state["saved_industry"] = industry
            st.session_state["saved_report"] = report
            st.session_state["saved_context"] = context
            st.session_state["saved_pages"] = [
                {"title": d.metadata.get("title", ""), "url": d.metadata.get("source", "")}
                for d in docs
            ]

            # reset chatbot state for new report
            st.session_state["chat_history"] = []
            st.session_state["last_answer_sources"] = []
            st.session_state["last_answer_evidence"] = ""
            st.session_state["report_ready"] = True

            # done
            st.session_state["run_state"] = "idle"

    except CancelRun:
        # Cancelled: just end the run and let UI be interactive
        st.session_state["run_state"] = "idle"

    finally:
        st.session_state["run_in_progress"] = False
        st.session_state["cancel_requested"] = False

        # If user queued a new industry during the run, start it automatically
        pending = (st.session_state.get("pending_industry") or "").strip()
        if pending:
            st.session_state["pending_industry"] = ""
            st.session_state["run_state"] = "running"
            st.session_state["run_industry"] = pending
            st.rerun()


# ============================================================
# Render report + PDF + chatbot whenever report_ready is True
# - This section is your existing rendering logic (kept intact)
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

        for m in st.session_state.get("chat_history", []):
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_q = st.chat_input("Ask a follow-up question about the industry…")

        if user_q:
            st.session_state["chat_history"].append({"role": "user", "content": user_q})

            chat_evidence, chat_sources = retrieve_chat_evidence(user_q, industry)

            st.session_state["last_answer_sources"] = chat_sources
            st.session_state["last_answer_evidence"] = chat_evidence

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