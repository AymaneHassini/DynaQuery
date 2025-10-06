# chains/dynamic_chains.py
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_google_genai import ChatGoogleGenerativeAI

from data.schema_utils import QueryPlan 
from config.settings import LLM_MODEL

llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")

def create_dynamic_sile_chain(schema_str: str):
    """
    Creates a SILE chain dynamically for a specific database schema.
    """
    prompt = f"""Return a query plan to answer the user's question.
The plan MUST identify the single `base_table` that contains the primary entity the user is asking about, and a list of `join_tables`.
Example: For "Find specs for Apple laptops", the plan is: base_table="specifications", join_tables=["products"].

The available tables are:
{schema_str}
"""
    return create_extraction_chain_pydantic(
        QueryPlan, 
        llm, 
        system_message=prompt
    )

def create_dynamic_rag_retriever(schema_str: str):
    """
    Builds a RAG retriever for a specific database schema string.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    schema_chunks = text_splitter.split_text(schema_str)  
    vector_store = Chroma.from_texts(texts=schema_chunks, embedding=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 4})