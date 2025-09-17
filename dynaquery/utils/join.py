# utils/join.py
"""
SQL JOIN generation utilities.
"""
from typing import List
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_langchain_llm
from prompts.templates import join_prompt

def create_join_chain():
    """Creates a LangChain for generating SQL JOIN clauses."""
    llm_join = get_langchain_llm()
    return join_prompt | llm_join | StrOutputParser()

def generate_left_join_query(filtered_schema: str, base_table: str, join_tables: List[str]) -> str:
    """
    Generate a candidate SQL query from an explicit query plan.
    
    Args:
        filtered_schema: Database schema filtered for the relevant tables.
        base_table: The single, explicit base table for the query.
        join_tables: A list of tables to LEFT JOIN to the base table.
        
    Returns:
        str: Generated SQL query with appropriate joins.
    """
    if not base_table:
        return ""
        
    if not join_tables:
        return f"SELECT * FROM {base_table};"
    
    join_chain = create_join_chain()
    
    join_input = {
        "schema": filtered_schema,
        "base_table": base_table,
        "join_tables": ", ".join(join_tables)
    }
    
    join_clauses = join_chain.invoke(join_input)
    
    if "no join" in join_clauses.strip().lower():
        return f"SELECT * FROM {base_table};"
    else:
        return f"SELECT * FROM {base_table} {join_clauses};"