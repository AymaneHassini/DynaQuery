# utils/sql.py
"""
SQL utility functions for query cleaning and processing.
"""
import re
from typing import List

def clean_sql_query(text: str) -> str:
    """
    Clean SQL query by removing code block syntax, various SQL tags, backticks,
    prefixes, and unnecessary whitespace while preserving the core SQL query.
    
    Args:
        text: Raw SQL query text, potentially with formatting
        
    Returns:
        str: Cleaned SQL query
    """
    # 1) Remove code block syntax
    block_pattern = r"```(?:sql|SQL|SQLQuery|mysql|postgresql)?\s*(.*?)\s*```"
    text = re.sub(block_pattern, r"\1", text, flags=re.DOTALL)

    # 2) Remove possible "SQLQuery:" prefix
    prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|MySQL|PostgreSQL|SQL)\s*:\s*"
    text = re.sub(prefix_pattern, "", text, flags=re.IGNORECASE)

    # 3) Extract the first SQL statement if there's random text after it
    sql_statement_pattern = r"(SELECT.*?;)"
    sql_match = re.search(sql_statement_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if sql_match:
        text = sql_match.group(1)

    # 4) Remove backticks around identifiers
    text = re.sub(r'`([^`]*)`', r'\1', text)

    # 5) Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # 6) Preserve newlines for main SQL keywords
    keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
                'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN',
                'OUTER JOIN', 'UNION', 'VALUES', 'INSERT', 'UPDATE', 'DELETE']

    pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
    text = re.sub(f'({pattern})', r'\n\1', text, flags=re.IGNORECASE)

    # 7) Final cleanup
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n', text)

    return text

def build_where_clause(pk_list: List[int], pk_col: str) -> str:
    """
    Build a SQL WHERE clause from a list of primary key values.
    
    Args:
        pk_list: List of primary key values
        pk_col: Name of the primary key column
        
    Returns:
        str: SQL WHERE clause with OR conditions
    """
    return " OR ".join(f"{pk_col} = {pk}" for pk in pk_list)