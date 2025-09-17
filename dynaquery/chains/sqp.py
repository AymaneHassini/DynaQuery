# chains/sqp.py

"""
Basic NL-to-SQL chain implementation.
"""
from langchain.memory import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from data.db_connector import  get_query_tool
from data.schema_utils import table_chain, get_table_details, filter_schema_for_tables
from models.llm import get_langchain_llm
from prompts.templates import create_zero_shot_prompt
from utils.sql import clean_sql_query
from chains.answer_chain import format_answer

def get_sqp_chain():
    """Create the zero-shot NL-to-SQL chain."""
    # 1) Get dependencies
    llm = get_langchain_llm()
    execute_query = get_query_tool()
    full_schema = get_table_details()
    
    # 2) Create final prompt (Zero-Shot)
    final_prompt = create_zero_shot_prompt()
    
    # 3) SQL query generator (NOW BUILT MANUALLY)
    # This is a standard LangChain Expression Language (LCEL) chain.
    # It takes the prompt, pipes it to the LLM, and then to a string parser.
    generate_query = final_prompt | llm | StrOutputParser()
    
    # 4) Clean query utility
    clean_query = RunnableLambda(clean_sql_query)
    
    # 5) Dynamic table extraction & filtering (SILE)
    sile_chain = (
        RunnablePassthrough.assign(
            # The table_chain now returns a QueryPlan object
            query_plan=lambda x: table_chain.invoke({"input": x["question"]})[0]
        )
        | RunnablePassthrough.assign(
            # We need to construct the list of all tables for filtering
            all_tables=lambda x: [x["query_plan"].base_table] + x["query_plan"].join_tables
        )
        | RunnablePassthrough.assign(
            filtered_schema=lambda x: filter_schema_for_tables(x["all_tables"])
        )
    )
    
    # 6) Final chain assembly
    chain = (
        sile_chain
        | RunnablePassthrough.assign(table_info=lambda x: x["filtered_schema"])
        | RunnablePassthrough.assign(
            # The input to generate_query must match the prompt's variables
            query=lambda x: generate_query.invoke({
                "table_info": x["table_info"],
                "input": x["input"],
                "messages": x["messages"]
            })
        )
        | RunnablePassthrough.assign(
            cleaned_query=lambda x: clean_query.invoke(x["query"])
        )
        | RunnablePassthrough.assign(
            result=lambda x: execute_query.run(x["cleaned_query"])
        )
        | (lambda x: format_answer(
            question=x["question"], 
            query=x["cleaned_query"], 
            result=x["result"]
        ))
    )
    
    return chain
    
def create_chat_history(messages):
    """
    Create a LangChain chat history from message list.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        ChatMessageHistory: LangChain chat history object
    """
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == "user":
            history.add_user_message(message["content"])
        else:
            history.add_ai_message(message["content"])
    return history

def invoke_sqp(question, messages):
    """Invoke the zero-shot NL-to-SQL chain."""
    chain = get_sqp_chain()
    history = create_chat_history(messages)
    try:

        response = chain.invoke({
            "question": question,
            "input": question,
            "messages": history.messages,
        })
    except IndexError:
        response = "I'm sorry, but I couldn't find any tables in the database that seem relevant to your question."
        
    history.add_user_message(question)
    history.add_ai_message(response)
    
    return response