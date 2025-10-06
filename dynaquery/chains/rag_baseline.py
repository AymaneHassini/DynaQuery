# chains/rag_baseline.py
"""
Implementation of the RAG-based baseline for schema linking.
This pipeline treats the schema as an unstructured document and uses
semantic search to find relevant tables.
"""
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from data.schema_utils import get_table_details

_rag_retriever = None

def get_rag_schema_retriever():
    """
    Builds and returns a RAG retriever for the database schema.
    Caches the retriever in a global variable to avoid re-building the index.
    """
    global _rag_retriever
    if _rag_retriever is not None:
        return _rag_retriever

    print("Building RAG schema retriever for the first time...")
    
    # 1. Get the entire database schema as a single text document.
    schema_document = get_table_details()

    # 2. Chunk the document. We use a recursive splitter which is a standard
    #    and robust choice for splitting code-like text.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""], # Splits by table, then by line
        chunk_size=512,
        chunk_overlap=100
    )
    schema_chunks = text_splitter.split_text(schema_document)
    
    print(f"Schema split into {len(schema_chunks)} chunks.")

    # 3. Embed the chunks and store them in a vector database.
    #    We use BAAI/llm-embedder, a model shown by Wang et al. (2024) to have
    #    a strong balance of performance and size for RAG tasks.
    print("Loading embedding model (BAAI/llm-embedder)...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
    
    # We use ChromaDB for its lightweight, file-based nature, which enhances reproducibility.
    vector_store = Chroma.from_texts(
        texts=schema_chunks,
        embedding=embedding_model,
        collection_name="schema_rag_collection"
    )

    # 4. Create the retriever object. We will retrieve the top 4 most relevant chunks.
    _rag_retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    print("RAG schema retriever built and cached successfully.")
    return _rag_retriever

def invoke_rag_for_linking(question: str) -> list[str]:
    """
    Invokes the RAG retriever for a given question and returns the list of
    predicted table names found in the retrieved context.
    """
    retriever = get_rag_schema_retriever()
    
    # Retrieve the most relevant schema chunks
    docs = retriever.invoke(question)
    retrieved_context = "\n\n".join(doc.page_content for doc in docs)
    
    # Parse the retrieved text to extract the table names
    # We use a regex to find all unique table names mentioned in the context.
    predicted_tables = list(set(re.findall(r"Table Name: (\w+)", retrieved_context)))
    
    return predicted_tables