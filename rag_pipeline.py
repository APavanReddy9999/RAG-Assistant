import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

# ── 1. LOAD EXISTING VECTOR STORE ──────────────────────────────────
def load_vector_store(persist_dir="./chroma_db"):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    print("✅ Vector store loaded from disk")
    return vector_store


# ── 2. CREATE RETRIEVER ─────────────────────────────────────────────
def create_retriever(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10}
    )
    return retriever


# ── 3. FORMAT RETRIEVED DOCS ────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ── 4. BUILD THE RAG CHAIN (Modern LCEL style) ──────────────────────
def build_rag_chain(retriever):
    prompt = PromptTemplate.from_template("""
    You are a helpful assistant. Use ONLY the context below to answer the question.
    If the answer is not in the context, say "I don't have enough information to answer this."
    Do NOT make up information.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

    # LCEL chain: retriever → prompt → llm → output parser
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ── 5. QUERY FUNCTION ───────────────────────────────────────────────
def ask(chain, retriever, question):
    print("-" * 60)

    answer = chain.invoke(question)
    print(f"💬 Answer:\n{answer}")

    # Show source documents
    docs = retriever.invoke(question)
    print("\n📄 Sources used:")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        print(f"  [{i}] {source} — Page {page}")


# ── MAIN ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    vector_store = load_vector_store()
    retriever = create_retriever(vector_store)
    chain = build_rag_chain(retriever)

    print("\n🤖 RAG Pipeline Ready! Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            break
        if question:
            ask(chain, retriever, question)