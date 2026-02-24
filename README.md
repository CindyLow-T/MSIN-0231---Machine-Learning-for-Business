📊 Market Research Assistant (Wikipedia-Grounded RAG System)
A production-style Retrieval-Augmented Generation (RAG) application that generates structured industry reports grounded strictly in Wikipedia evidence, with dynamic follow-up Q&A.
Built using Streamlit, OpenAI API, and LangChain WikipediaRetriever.

**🚀 Overview**
This system transforms a simple industry query into:
✅ A structured, sub-500 word industry report
✅ Evidence-grounded content (Wikipedia-only)
✅ Downloadable professional PDF
✅ Interactive follow-up Q&A chatbot
✅ Strict hallucination controls
The architecture prioritises groundedness, robustness, and UX stability over raw generative output.

**🧠 System Architecture**
User Input
→ Industry Intent Gate
→ Wikipedia Retrieval (Top-K)
→ Industry-Level Page Filtering
→ Text Chunking
→ Embedding + Cosine Reranking
→ LLM Report Generation (Grounded)
→ PDF Export
→ Wikipedia-Only Chatbot (Dynamic Retrieval Per Question)

**🔎 Key Design Features**
**1️⃣ Industry-Intent Gate**
Prevents non-industry queries from triggering expensive API calls.
Blocks random names and nonsensical inputs
Handles edge cases (e.g., “kpop industry”, hyphen variants)
Reduces wasted embedding and LLM usage
**2️⃣ Dual Retrieval Strategy**
Report Pipeline
Strict filtering for industry-level overview pages
Removes overly specific titles (films, albums, characters, etc.)
Chatbot Pipeline
Broader retrieval scope
Only excludes Wikipedia meta/disambiguation pages
Reduces false “unrelated” responses
**3️⃣ Evidence-Grounded RAG**
Wikipedia pages chunked with overlap
Embeddings generated via text-embedding-3-small
Cosine similarity reranking
LLM restricted to retrieved evidence only
Explicit instruction to avoid hallucination
**4️⃣ UX Stability (Streamlit State Management)**
Persistent session_state for reports and chat
No nested expanders
PDF download does not reset UI
API key validation before pipeline execution
**5️⃣ Performance Optimisation**
Embedding caching with @st.cache_data
Stable hashing for cache keys
Hard caps on chunk counts
Batched embedding requests

**🛠 Tech Stack**
Frontend: Streamlit
LLM: OpenAI gpt-4o-mini
Embeddings: text-embedding-3-small
Retriever: LangChain WikipediaRetriever
Vector Ranking: NumPy (cosine similarity)
PDF Export: ReportLab Platypus

**📦 Installation**
pip install -r requirements.txt
streamlit run app.py

**🔐 Environment Requirements**
An OpenAI API key is required.
The application includes built-in key validation.

**📈 Performance Considerations**
First-run queries are slower due to embedding generation
Wikipedia retrieval latency depends on external API responsiveness
Streamlit executes synchronously (no true mid-run cancellation)

**⚠ Limitations**
Wikipedia-only data source (no proprietary databases)
No persistent vector database (in-memory caching only)
Synchronous execution model
Limited observability and logging

**🏢 Enterprise Upgrade Path**
For production deployment at scale:
Replace in-memory caching with Redis
Store embeddings in a vector database (e.g., pgvector / Pinecone)
Implement async task queue for long-running operations
Add structured logging + monitoring
Integrate approved internal data sources
Enforce role-based access control

**📄 Example Use Cases**
Rapid industry brief generation
Pre-meeting executive summaries
Market orientation research
Structured Q&A exploration

**👩‍💻 Author**
Cindy Low
MSc Business Analytics
