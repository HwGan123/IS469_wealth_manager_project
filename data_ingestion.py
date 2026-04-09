from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def build_vector_db(pdf_files: list):
    # 1. Load and Split
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    for pdf in pdf_files:
        loader = PyPDFLoader(pdf)
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)
    
    # 2. Create ChromaDB with FREE Embeddings
    Chroma.from_documents(
        documents=all_chunks, 
        embedding=hf_embeddings, # Using the HF model defined above
        persist_directory="./chroma_db"
    )
    print("✅ Local Vector Database Created successfully.")