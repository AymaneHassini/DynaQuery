# data/schema_utils.py
"""
Database schema utilities for extracting and processing schema information.
This file contains the core logic for the Schema Introspection and Linking Engine (SILE).
"""
import streamlit as st
from typing import List
from operator import itemgetter
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from pydantic.v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from dynaquery.config.settings import DB_USER, DB_PASSWORD, DB_HOST, DB_NAME, LLM_MODEL
from dynaquery.data.db_connector import get_inspector
import json

# Make sure environment variables are not None
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_NAME]):
    raise ValueError("Missing one or more DB environment variables in .env (DB_USER, DB_PASSWORD, DB_HOST, DB_NAME).")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

class QueryPlan(BaseModel):
    """A plan for executing a query, including the base table and any tables to join."""
    base_table: str = Field(description="The single table that contains the primary entity the user is asking for.")
    join_tables: List[str] = Field(description="A list of any other tables that need to be joined to the base table to answer the query.")

def load_schema_comments():
    """Loads the semantic schema comments from a JSON file."""
    try:
        with open('config/schema_comments.json', 'r') as f: 
            return json.load(f)
    except FileNotFoundError:
        print("WARNING: schema_comments.json not found. Proceeding with raw schema.")
        return {"tables": {}, "columns": {}}
    except json.JSONDecodeError:
        print("WARNING: Could not parse schema_comments.json. Proceeding with raw schema.")
        return {"tables": {}, "columns": {}}

@st.cache_data
def get_table_details():
    """
    Extract table details from the database, enriches them with semantic comments,
    and formats as a string.
    
    Returns:
        A formatted, semantically enriched string containing details of all tables.
    """
    schema_comments = load_schema_comments()
    table_comments = schema_comments.get("tables", {})
    column_comments = schema_comments.get("columns", {})
    
    inspector = get_inspector()
    table_names = inspector.get_table_names()
    metadata_str = ""
    
    for table_name in table_names:
        table_comment = table_comments.get(table_name, "")
        if table_comment:
            metadata_str += f"Table Name: {table_name} -- {table_comment}\nColumns:\n"
        else:
            metadata_str += f"Table Name: {table_name}\nColumns:\n"
        
        # Get column details
        columns = inspector.get_columns(table_name)
        for col in columns:
            col_name = col["name"]
            col_type = str(col["type"])
            is_nullable = col["nullable"]
            
            fully_qualified_col_name = f"{table_name}.{col_name}"
            col_comment = column_comments.get(fully_qualified_col_name, "")
            
            metadata_str += f" - {col_name} ({col_type}), nullable={is_nullable}"
            if col_comment:
                metadata_str += f" -- {col_comment}\n"
            else:
                metadata_str += "\n"
        
        # Get primary key details 
        pk = inspector.get_pk_constraint(table_name)
        pk_columns = ", ".join(pk.get("constrained_columns", []))
        metadata_str += f"Primary Key Columns: {pk_columns if pk_columns else 'None'}\n"
        
        # Get foreign key details 
        fks = inspector.get_foreign_keys(table_name)
        if fks:
            metadata_str += "Foreign Keys:\n"
            for fk in fks:
                fk_cols = ", ".join(fk.get("constrained_columns", []))
                referred_table = fk.get("referred_table", "Unknown")
                referred_cols = ", ".join(fk.get("referred_columns", []))
                metadata_str += f" - {fk_cols} -> {referred_table}({referred_cols})\n"
        else:
            metadata_str += "Foreign Keys: None\n"
            
        metadata_str += "\n"
        
    return metadata_str

def filter_schema_for_tables(selected_tables: List[str]) -> str:
    """
    Filter the full schema to include only selected tables.
    
    Args:
        selected_tables: List of table names to include
        
    Returns:
        Filtered schema containing only the selected tables
    """
    if not selected_tables:
        return table_details # Return the full schema if no tables are selected
        
    return "\n\n".join(
        section for section in table_details.split("\n\n")
        if any(f"Table Name: {table}" in section for table in selected_tables)
    )

# Get the database schema once at module load time
table_details = get_table_details()

table_details_prompt = f"""Return a query plan to answer the user's question.
The plan MUST identify the single `base_table` that contains the primary entity the user is asking about, and a list of `join_tables` needed for filtering or additional information.

Example: For the query "Find all specifications for Apple laptops", the primary entity is "specifications". The `products` table is needed for filtering.
So, the plan would be: base_table="specifications", join_tables=["products"].

The available tables are:
{table_details}
"""

table_chain = create_extraction_chain_pydantic(
    QueryPlan, 
    llm, 
    system_message=table_details_prompt
)