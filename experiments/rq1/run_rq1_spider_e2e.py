# run_rq1_spider_e2e.py
"""
Definitive, self-contained evaluation script for Research Question 3 (End-to-End).
This script compares the end-to-end Execution Accuracy of a Text-to-SQL pipeline
when using our SILE vs. a RAG baseline for schema linking.
"""

import os
import json
from tqdm import tqdm
import re
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dynaquery.config.settings import LLM_MODEL
from dynaquery.prompts.templates import create_zero_shot_prompt
from dynaquery.utils.sql import clean_sql_query

# --- 2. Configuration ---
OFFICIAL_SPIDER_DEV_PATH = "external_data/spider_data/dev.json"
OFFICIAL_SPIDER_TABLES_PATH = "external_data/spider_data/tables.json"
SAMPLE_PATH = "outputs/spider/spider_linking_sample_500.jsonl"

# --- 3. Re-implement Chains Locally for Controlled Experiment ---

class QueryPlan(BaseModel):
    """A plan for executing a query, including the base table and any tables to join."""
    base_table: str = Field(description="The single table that contains the primary entity the user is asking for.")
    join_tables: list[str] = Field(description="A list of any other tables that need to be joined to the base table to answer the query.")

# Initialize models once to be reused globally within this script
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")

def create_dynamic_sile_chain(schema_str: str):
    """Creates a SILE chain dynamically for a specific database schema."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at creating query plans.
Your goal is to return a query plan to answer the user's question based on the provided schema.
The plan MUST identify the single `base_table` that contains the primary entity the user is asking about, and a list of `join_tables` needed for filtering or additional information.
Example: For "Find specs for Apple laptops", the plan is: base_table="specifications", join_tables=["products"].

The available tables are:
{schema_str}
"""),
        ("human", "{input}")
    ])
    structured_llm = llm.with_structured_output(QueryPlan)
    return prompt | structured_llm

def create_dynamic_rag_retriever(schema_str: str):
    """Builds a RAG retriever for a specific database schema string."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    schema_chunks = text_splitter.split_text(schema_str)
    vector_store = Chroma.from_texts(texts=schema_chunks, embedding=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 4})

def get_local_sql_generation_chain():
    """
    Creates the core ZERO-SHOT SQL generation pipeline, isolated for this experiment.
    """
    prompt = create_zero_shot_prompt()
    sql_chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(clean_sql_query)
    )
    return sql_chain


# --- 4. Data Loading and Schema Formatting ---
def load_spider_data():
    """Loads the official Spider data and creates lookup maps."""
    print(f"Loading official Spider data...")
    with open(OFFICIAL_SPIDER_DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(OFFICIAL_SPIDER_TABLES_PATH, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    schema_map = {db['db_id']: db for db in tables_data}
    question_map = {" ".join(entry['question'].split()): entry for entry in dev_data}
    print(f"Successfully built lookup maps.")
    return question_map, schema_map

def format_schema_for_prompt(schema_info: dict) -> str:
    """Formats the structured schema from tables.json into a string."""
    schema_str = ""
    table_names = schema_info.get('table_names_original', [])
    column_names = schema_info.get('column_names_original', [])
    
    for i, table_name in enumerate(table_names):
        schema_str += f"Table Name: {table_name}\nColumns:\n"
        for col_idx, col_name in column_names:
            if col_idx == i:
                schema_str += f" - {col_name}\n"
        schema_str += "\n"
    return schema_str.strip()

def filter_schema_for_tables(full_schema: str, selected_tables: list[str]) -> str:
    """A local implementation of the schema filtering utility."""
    if not selected_tables:
        return full_schema
    
    selected_tables_norm = {table.lower() for table in selected_tables}
    
    filtered_sections = []
    for section in full_schema.split("\n\n"):
        match = re.search(r"Table Name: (\w+)", section)
        if match:
            table_name = match.group(1).lower()
            if table_name in selected_tables_norm:
                filtered_sections.append(section)
    return "\n\n".join(filtered_sections)

# --- 5. Main Experiment Script ---
def run_e2e_generation():
    """Runs the full end-to-end evaluation and generates prediction files."""
    
    question_map, schema_map = load_spider_data()
    
    try:
        with open(SAMPLE_PATH, 'r', encoding='utf-8') as f:
            test_set_questions = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: Sample file not found at {SAMPLE_PATH}.")
        return

    sile_predictions = []
    rag_predictions = []
    gold_queries_for_eval = []

    print(f"\nRunning End-to-End SQL Generation on {len(test_set_questions)} samples...")
    
    sql_generation_chain = get_local_sql_generation_chain()

    for entry in tqdm(test_set_questions, desc="Generating SQL"):
        question = " ".join(entry['text'].split())
        
        official_entry = question_map.get(question)
        if not official_entry:
            print(f"Warning: Skipping question not found: '{question}'")
            continue
            
        db_id = official_entry['db_id']
        gold_sql = official_entry['query']
        schema_info = schema_map[db_id]
        schema_str = format_schema_for_prompt(schema_info)
        
        gold_queries_for_eval.append(f"{gold_sql}\t{db_id}")

        # --- Evaluate DynaQuery (SILE) Pipeline ---
        try:
            sile_linker = create_dynamic_sile_chain(schema_str)
            query_plan = sile_linker.invoke({"input": question})
            all_tables = list(set([query_plan.base_table] + query_plan.join_tables))
            filtered_schema = filter_schema_for_tables(schema_str, all_tables)
            
            generated_sql_sile = sql_generation_chain.invoke({
                "input": question,
                "table_info": filtered_schema,
                "messages": []
            })
            sile_predictions.append((generated_sql_sile, db_id))
        except Exception as e:
            print(f"Error in SILE pipeline for '{question}': {e}")
            sile_predictions.append(("SELECT 'ERROR'", db_id))

        # --- Evaluate RAG Baseline Pipeline ---
        try:
            rag_retriever = create_dynamic_rag_retriever(schema_str)
            docs = rag_retriever.invoke(question)
            rag_context = "\n\n".join(doc.page_content for doc in docs)

            generated_sql_rag = sql_generation_chain.invoke({
                "input": question,
                "table_info": rag_context,
                "messages": []
            })
            rag_predictions.append((generated_sql_rag, db_id))
        except Exception as e:
            print(f"Error in RAG pipeline for '{question}': {e}")
            rag_predictions.append(("SELECT 'ERROR'", db_id))
    # --- Write Output Files ---
    output_dir = "outputs/spider"
    # The main linking script already creates this, but this makes the script runnable on its own.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving prediction and gold files to '{output_dir}' directory...")
    # Define full file paths
    dynaquery_pred_path = os.path.join(output_dir, "predictions_dynaquery-schema-linking.sql")
    rag_pred_path = os.path.join(output_dir, "predictions_rag-schema-linking.sql")
    gold_sample_path = os.path.join(output_dir, "dev_gold_sample-schema-linking.sql")

    # Write the prediction and gold files
    with open(dynaquery_pred_path, "w", encoding='utf-8') as f:
        for sql, db_id in sile_predictions:
            f.write(f"{sql}\t{db_id}\n")
            
    with open(rag_pred_path, "w", encoding='utf-8') as f:
        for sql, db_id in rag_predictions:
            f.write(f"{sql}\t{db_id}\n")
            
    with open(gold_sample_path, "w", encoding='utf-8') as f:
        f.write("\n".join(gold_queries_for_eval))

    print("\n" + "="*50)
    print("Prediction and gold files generated successfully.")
    print("="*50)
if __name__ == "__main__":
    run_e2e_generation()