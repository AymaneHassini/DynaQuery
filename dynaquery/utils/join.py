# utils/join.py
"""
SQL JOIN generation utilities.
"""
from typing import List
from langchain_core.output_parsers import StrOutputParser

from dynaquery.models.llm import get_langchain_llm
from dynaquery.prompts.templates import join_prompt

def create_join_chain():
    """Creates a LangChain for generating SQL JOIN clauses."""
    llm_join = get_langchain_llm()
    return join_prompt | llm_join | StrOutputParser()

def generate_join_clauses(filtered_schema: str, base_table: str, join_tables: List[str]) -> str:
    """
    Generates the FROM and JOIN clauses for a SQL query.
    
    Returns:
        str: A string like "FROM base_table LEFT JOIN table2 ON ..."
    """
    if not base_table:
        return ""
        
    # Start with the FROM clause
    from_and_joins = f"FROM {base_table}"
    
    if not join_tables:
        return from_and_joins
    
    join_chain = create_join_chain()
    
    join_input = {
        "schema": filtered_schema,
        "base_table": base_table,
        "join_tables": ", ".join(join_tables)
    }
    
    join_clauses_str = join_chain.invoke(join_input)
    
    if "no join" not in join_clauses_str.strip().lower():
        # Append the generated JOIN clauses
        from_and_joins += f" {join_clauses_str}"
        
    return from_and_joins