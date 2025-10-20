# chains/answer_chain.py
"""
Answer formatting chain for providing human-friendly responses.
"""
from langchain_core.output_parsers import StrOutputParser

from dynaquery.models.llm import get_langchain_llm
from dynaquery.prompts.templates import answer_prompt

def get_answer_chain():
    """
    Create a chain for generating user-friendly answers from SQL results.
    
    Returns:
        A runnable chain that formats SQL results into natural language
    """
    llm = get_langchain_llm()
    
    # Build and return the answer chain
    return answer_prompt | llm | StrOutputParser()

def format_answer(question: str, query: str, result: str, table_info: str = "") -> str:
    """
    Format SQL results into a user-friendly answer.
    
    Args:
        question: User's original question
        query: SQL query that was executed
        result: Raw SQL result string
        table_info: Optional table schema information
        
    Returns:
        str: User-friendly answer based on the SQL results
    """
    chain = get_answer_chain()
    return chain.invoke({
        "question": question,
        "query": query,
        "result": result
    })