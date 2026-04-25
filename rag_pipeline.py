"""
rag_pipeline.py — with Guardrails (clean terminal output)

WHAT CHANGED:
  1. HTTP request logs suppressed (httpx, httpcore, urllib3)
  2. Guard logs go to file only (not terminal)
  3. Terminal shows ONLY: question separator, answer, sources
"""

import os
import logging
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from guardrails import run_input_guards, run_output_guards

load_dotenv()

# ── Silence noisy HTTP and library loggers ────────────────────────
# WHY: httpx logs every API call like "HTTP Request: POST https://..."
# These are internal implementation details the user doesn't need.
# We silence them so the terminal stays clean.
# The guardrails.log file still captures security audit events.
for noisy_logger in [
    "httpx", "httpcore", "urllib3",
    "langchain", "langchain_core", "langchain_groq",
    "langchain_google_genai", "chromadb",
    "openai", "groq",
]:
    logging.getLogger(noisy_logger).setLevel(logging.ERROR)


# ── 1. LOAD VECTOR STORE ──────────────────────────────────────────
def load_vector_store(persist_dir="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("✅ Vector store loaded")
    return vector_store


# ── 2. CREATE RETRIEVER ───────────────────────────────────────────
def create_retriever(vector_store):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )


# ── 3. FORMAT DOCS ────────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── 4. BUILD RAG CHAIN ────────────────────────────────────────────
def build_rag_chain(retriever):
    prompt = PromptTemplate.from_template("""
You are a helpful healthcare assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."
Do NOT make up information. Do NOT share any personal patient data.

Context:
{context}

Question: {question}

Answer:""")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ── 5. ASK — clean terminal output ───────────────────────────────
def ask(chain, retriever, question):
    print("\n" + "─" * 60)

    # ── INPUT GUARD ───────────────────────────────────────────────
    input_result = run_input_guards(question)
    if not input_result.passed:
        print(f"\n{input_result.message}\n")
        return

    # ── PIPELINE ──────────────────────────────────────────────────
    answer = chain.invoke(question)

    # ── OUTPUT GUARD ──────────────────────────────────────────────
    output_result = run_output_guards(answer)
    if not output_result.passed:
        print(f"\n{output_result.message}\n")
        return

    # ── CLEAN OUTPUT ──────────────────────────────────────────────
    print(f"\n💬 Answer:\n{answer}\n")

    docs = retriever.invoke(question)
    print("📄 Sources:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page   = doc.metadata.get('page', 'N/A')
        print(f"  [{i}] {source} — Page {page}")


# ── MAIN ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    vector_store = load_vector_store()
    retriever    = create_retriever(vector_store)
    chain        = build_rag_chain(retriever)

    print("\n🤖 Healthcare RAG Assistant (Guardrails ON)")
    print("   Type 'exit' to quit\n")

    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            break
        if question:
            ask(chain, retriever, question)