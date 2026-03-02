import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   # ← updated
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# ── 1. LOAD DOCUMENTS ──────────────────────────────────────────────
def load_documents(docs_path="./docs"):
    """
    Loads all PDFs and TXT files from the docs/ folder.
    DirectoryLoader handles multiple files automatically.
    """
    loaders = [
        DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_path, glob="**/*.txt", loader_cls=TextLoader),
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    print(f"✅ Loaded {len(documents)} document pages/chunks")
    return documents


# ── 2. SPLIT INTO CHUNKS ────────────────────────────────────────────
def split_documents(documents):
    """
    Splits large documents into smaller overlapping chunks.
    
    chunk_size=1000    → each chunk has ~1000 characters
    chunk_overlap=200  → 200 chars overlap between chunks 
                         (so context isn't lost at boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # tries to split at paragraphs first
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


# ── 3. CREATE EMBEDDINGS & STORE IN CHROMADB ────────────────────────
def create_vector_store(chunks, persist_dir="./chroma_db"):
    """
    Converts each chunk into a vector (list of numbers) using Gemini 
    embeddings, then stores everything in ChromaDB on your local disk.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
    
    # Chroma.from_documents() embeds + stores in one call
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir  # saves to disk so you don't re-embed every run
    )
    
    print(f"✅ Vector store created and saved to '{persist_dir}'")
    return vector_store


# ── RUN INGESTION ───────────────────────────────────────────────────
if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    vector_store = create_vector_store(chunks)
    print("\n🎉 Ingestion complete! You can now run rag_pipeline.py")