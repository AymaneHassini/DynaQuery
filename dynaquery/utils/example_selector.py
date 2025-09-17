# In utils/example_selector.py

from langchain_community.vectorstores import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
import json

# We will cache the vector store, which is the expensive part to build.
_vector_store_singleton = None

def get_spider_example_selector(embedding_model):
    """
    Builds and returns a few-shot example selector for the Spider dataset.
    This function caches the expensive-to-build vector store to avoid re-computation,
    but returns a new selector instance to prevent state corruption issues.
    """
    global _vector_store_singleton

    # Build the vector store only once.
    if _vector_store_singleton is None:
        print("Building the few-shot example vector store for the first time...")
        
        SPIDER_TRAIN_PATH = "/Users/aymenhassini/Downloads/spider_data/train_spider.json" 
        
        try:
            with open(SPIDER_TRAIN_PATH, "r", encoding='utf-8') as f:
                train_data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: The Spider training file was not found at '{SPIDER_TRAIN_PATH}'")
            raise

        # The format LangChain expects for examples
        examples = [
            {"input": item["question"], "query": item["query"]}
            for item in train_data
        ]

        # --- THE FIX IS HERE ---
        # We need to extract the texts (questions) to be embedded.
        example_texts = [example["input"] for example in examples]
        
        # The correct method is Chroma.from_texts.
        # We pass the full examples as metadata to retrieve them later.
        _vector_store_singleton = Chroma.from_texts(
            texts=example_texts,
            embedding=embedding_model,
            metadatas=examples  # Store the full example dict in metadata
        )
        print("Vector store for example selection built successfully.")

    # Always create and return a NEW selector instance.
    # This is lightweight and points to our heavy, cached vector store.
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=_vector_store_singleton,
        k=3,
        input_keys=["input"] # Tell the selector which key from the chain's input to use for the search
    )
    
    return example_selector