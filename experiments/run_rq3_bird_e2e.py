# run_rq3_bird_e2e.py
"""
Definitive, FINAL, self-contained evaluation script for the BIRD benchmark.
This script runs a reproducible, ROBUST STRATIFIED random sample of 500 questions.

It compares the end-to-end Execution Accuracy of a Text-to-SQL pipeline
when using our SILE vs. a RAG baseline for schema linking on the BIRD dataset.

It generates predictions in the official JSON dictionary format required by BIRD's
evaluation.py and evaluation_ves.py scripts, with verbose debugging output.
"""
import os
import json
import random
from tqdm import tqdm
import re
from collections import defaultdict
import math

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dynaquery.config.settings import LLM_MODEL
from dynaquery.utils.sql import clean_sql_query

# --- 2. Configuration ---
BIRD_DEV_PATH = "external_data/bird/dev.json"
BIRD_TABLES_PATH = "external_data/bird/dev_tables.json"
SAMPLE_SIZE = 500
BIRD_SAMPLE_PATH = "data_samples/bird_dev_sample_500.json"
# --- 3. Re-implement Chains Locally for Controlled Experiment ---
class QueryPlan(BaseModel):
    """A plan for executing a query, including the base table and any tables to join."""
    base_table: str = Field(description="The single table that contains the primary entity the user is asking for.")
    join_tables: list[str] = Field(description="A list of any other tables that need to be joined to the base table to answer the query.")

# Initialize models once to be reused globally
print("Initializing LLM and Embedding Model...")
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/llm-embedder")
print("Models initialized.")

def create_dynamic_sile_chain(schema_str: str):
    from langchain.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert at creating query plans...
The available tables are:\n{schema_str}"""),
        ("human", "{input}")
    ])
    structured_llm = llm.with_structured_output(QueryPlan)
    return prompt | structured_llm

def create_dynamic_rag_retriever(schema_str: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    schema_chunks = text_splitter.split_text(schema_str)
    vector_store = Chroma.from_texts(texts=schema_chunks, embedding=embedding_model)
    return vector_store.as_retriever(search_kwargs={"k": 4})

def get_bird_sql_generation_chain():
    bird_prompt_template = """You are an expert SQL developer...
### Database Schema:
{table_info}
### Evidence:
{evidence}
### Question:
{input}
### SQL Query:
"""
    prompt = PromptTemplate(
        template=bird_prompt_template,
        input_variables=["table_info", "evidence", "input"]
    )
    sql_chain = (prompt | llm | StrOutputParser() | RunnableLambda(clean_sql_query))
    return sql_chain

# --- 4. Data Loading, Sampling, and Schema Formatting ---

def create_bird_stratified_sample():
    """
    Creates a reproducible, robust stratified random sample from the BIRD dev set,
    correctly handling remainders to ensure an exact sample size.
    """
    print(f"Creating a stratified random sample of {SAMPLE_SIZE} from {BIRD_DEV_PATH}...")
    try:
        with open(BIRD_DEV_PATH, 'r', encoding='utf-8') as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: The full BIRD dataset was not found at '{BIRD_DEV_PATH}'")
        return False

    # Group data by difficulty
    by_difficulty = defaultdict(list)
    for item in full_data:
        by_difficulty[item['difficulty']].append(item)

    # --- STRATIFICATION LOGIC ---
    total_size = len(full_data)
    final_sample = []
    random.seed(42) # for reproducibility

    # Step 1: Calculate initial counts (floor) and fractional remainders
    proportions = {key: len(val) / total_size for key, val in by_difficulty.items()}
    
    # Calculate the exact number of samples needed (float)
    exact_counts = {key: prop * SAMPLE_SIZE for key, prop in proportions.items()}
    
    # Get the integer part and the fractional part for each group
    final_counts = {key: math.floor(count) for key, count in exact_counts.items()}
    remainders = {key: count - final_counts[key] for key, count in exact_counts.items()}

    # Step 2: Calculate how many samples are missing due to flooring
    num_missing = SAMPLE_SIZE - sum(final_counts.values())

    # Step 3: Distribute missing samples to groups with the largest fractional remainders
    groups_by_remainder = sorted(remainders.keys(), key=lambda k: remainders[k], reverse=True)
    
    for i in range(num_missing):
        group_to_add_to = groups_by_remainder[i]
        final_counts[group_to_add_to] += 1

    # Step 4: Perform the final sampling with the corrected counts
    for difficulty, num_to_sample in final_counts.items():
        items = by_difficulty[difficulty]
        print(f"Sampling {num_to_sample} items from '{difficulty}' group (size {len(items)})")
        
        # Safety check: ensure we don't try to sample more items than exist
        if num_to_sample > len(items):
            print(f"Warning: Not enough items in '{difficulty}' to sample {num_to_sample}. Sampling all {len(items)}.")
            num_to_sample = len(items)
        final_sample.extend(random.sample(items, num_to_sample))

    # Final shuffle of the combined sample
    random.shuffle(final_sample)

    # Final assertion to guarantee correctness
    assert len(final_sample) == SAMPLE_SIZE, f"CRITICAL ERROR: Final sample size is {len(final_sample)}, expected {SAMPLE_SIZE}"

    os.makedirs(os.path.dirname(BIRD_SAMPLE_PATH), exist_ok=True)
    with open(BIRD_SAMPLE_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_sample, f, indent=4)

    print(f"Successfully created robust stratified sample file at {BIRD_SAMPLE_PATH} with exactly {len(final_sample)} items.")
    return True

def load_bird_data():
    print(f"Loading official BIRD data...")
    with open(BIRD_DEV_PATH, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(BIRD_TABLES_PATH, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    schema_map = {db['db_id']: db for db in tables_data}
    print(f"Successfully loaded {len(dev_data)} questions and {len(schema_map)} schemas.")
    return dev_data, schema_map

def format_schema_for_prompt(schema_info: dict) -> str:
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
    if not selected_tables: return full_schema
    selected_tables_norm = {table.lower() for table in selected_tables}
    filtered_sections = []
    for section in full_schema.split("\n\n"):
        match = re.search(r"Table Name: (\w+)", section)
        if match and match.group(1).lower() in selected_tables_norm:
            filtered_sections.append(section)
    return "\n\n".join(filtered_sections)

# --- 5. Main Experiment Script ---
def run_bird_e2e_generation():
    full_dev_data, schema_map = load_bird_data()

    # Load or create the stratified sample
    try:
        with open(BIRD_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    except FileNotFoundError:
        if not create_bird_stratified_sample(): return
        with open(BIRD_SAMPLE_PATH, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
    
    # Create a set of question_ids for fast checking
    sample_ids = {item['question_id'] for item in sample_data}
    print(f"\nRunning End-to-End SQL Generation on a sample of {len(sample_ids)} BIRD questions...")

    sile_predictions_dict = {}
    rag_predictions_dict = {}
    sql_generation_chain = get_bird_sql_generation_chain()

    # Loop over the FULL dataset to preserve indices for the evaluation script
    for i, entry in enumerate(tqdm(full_dev_data, desc="Processing BIRD dev set")):
        db_id = entry['db_id']
        
        # Only run the expensive pipelines for items in our sample
        if entry['question_id'] in sample_ids:
            question = entry['question']
            evidence = entry['evidence']
            
            print(f"\n{'='*20} Processing Sample (ID: {entry['question_id']}, Difficulty: {entry['difficulty']}) {'='*20}")
            print(f"[DEBUG] Question: {question}")

            schema_info = schema_map.get(db_id)
            if not schema_info:
                print(f"Warning: Skipping question for missing db_id: '{db_id}'")
                sile_predictions_dict[str(i)] = f"SELECT 'SKIPPED_NO_SCHEMA'\t----- bird -----\t{db_id}"
                rag_predictions_dict[str(i)] = f"SELECT 'SKIPPED_NO_SCHEMA'\t----- bird -----\t{db_id}"
                continue
            
            schema_str = format_schema_for_prompt(schema_info)
            
            # --- Evaluate DynaQuery (SILE) Pipeline ---
            try:
                sile_linker = create_dynamic_sile_chain(schema_str)
                query_plan = sile_linker.invoke({"input": question})
                all_tables = list(set([query_plan.base_table] + query_plan.join_tables))
                filtered_schema = filter_schema_for_tables(schema_str, all_tables)
                
                print(f"[DEBUG-SILE] Linked tables: {all_tables}")
                # print(f"[DEBUG-SILE] Pruned schema context:\n---\n{filtered_schema}\n---") # Optional: uncomment for very verbose output

                generated_sql_sile = sql_generation_chain.invoke({
                    "input": question, "table_info": filtered_schema, "evidence": evidence
                })
                print(f"[DEBUG-SILE] Generated SQL: {generated_sql_sile}")
                sile_output_string = f"{generated_sql_sile}\t----- bird -----\t{db_id}"
                sile_predictions_dict[str(i)] = sile_output_string
            except Exception as e:
                print(f"ERROR in SILE pipeline for '{question}': {e}")
                sile_predictions_dict[str(i)] = f"SELECT 'ERROR'\t----- bird -----\t{db_id}"

            # --- Evaluate RAG Baseline Pipeline ---
            try:
                rag_retriever = create_dynamic_rag_retriever(schema_str)
                docs = rag_retriever.invoke(question)
                rag_context = "\n\n".join(doc.page_content for doc in docs)
                # print(f"[DEBUG-RAG] Retrieved context:\n---\n{rag_context}\n---") # Optional: uncomment for very verbose output
                generated_sql_rag = sql_generation_chain.invoke({
                    "input": question, "table_info": rag_context, "evidence": evidence
                })
                print(f"[DEBUG-RAG] Generated SQL: {generated_sql_rag}")
                rag_output_string = f"{generated_sql_rag}\t----- bird -----\t{db_id}"
                rag_predictions_dict[str(i)] = rag_output_string
            except Exception as e:
                print(f"ERROR in RAG pipeline for '{question}': {e}")
                rag_predictions_dict[str(i)] = f"SELECT 'ERROR'\t----- bird -----\t{db_id}"
        
        else: # If not in sample, add a placeholder
            sile_predictions_dict[str(i)] = f"SELECT 'SKIPPED_NOT_IN_SAMPLE'\t----- bird -----\t{db_id}"
            rag_predictions_dict[str(i)] = f"SELECT 'SKIPPED_NOT_IN_SAMPLE'\t----- bird -----\t{db_id}"

    # --- Write Output Files ---
    output_dir_dynaquery = "outputs/bird_dynaquery/"
    output_dir_rag = "outputs/bird_rag/"
    os.makedirs(output_dir_dynaquery, exist_ok=True)
    os.makedirs(output_dir_rag, exist_ok=True)

    with open(os.path.join(output_dir_dynaquery, "predict_dev.json"), "w", encoding='utf-8') as f:
        json.dump(sile_predictions_dict, f, indent=4)
            
    with open(os.path.join(output_dir_rag, "predict_dev.json"), "w", encoding='utf-8') as f:
        json.dump(rag_predictions_dict, f, indent=4)
    
    print("\n" + "="*50)
    print(f"Prediction files for BIRD generated based on a {len(sample_ids)}-item sample.")
    print(f"DynaQuery predictions are in: {output_dir_dynaquery}predict_dev.json")
    print(f"RAG predictions are in: {output_dir_rag}predict_dev.json")
    print("Next, run the official BIRD evaluation scripts.")
    print("="*50)

if __name__ == "__main__":
    run_bird_e2e_generation()