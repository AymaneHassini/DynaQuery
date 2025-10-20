# chains/sqp.py

"""
Basic NL-to-SQL chain implementation.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
    """
    Creates the core SQP chain that generates and executes a SQL query.
    It returns a dictionary of all intermediate steps.
    """
    llm = get_langchain_llm()
    execute_query = get_query_tool()
    
    final_prompt = create_zero_shot_prompt()
    generate_query = final_prompt | llm | StrOutputParser()
    clean_query = RunnableLambda(clean_sql_query)
    
    sile_chain = (
        RunnablePassthrough.assign(
            query_plan=lambda x: table_chain.invoke({"input": x["question"]})[0]
        )
        | RunnablePassthrough.assign(
            all_tables=lambda x: [x["query_plan"].base_table] + x["query_plan"].join_tables
        )
        | RunnablePassthrough.assign(
            filtered_schema=lambda x: filter_schema_for_tables(x["all_tables"])
        )
    )
    
    chain = (
        sile_chain
        | RunnablePassthrough.assign(table_info=lambda x: x["filtered_schema"])
        | RunnablePassthrough.assign(
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

def invoke_sqp(question, messages, return_dict=False):
    """
    Invoke the zero-shot NL-to-SQL chain.
    Can return either a formatted string (for UI) or a result dictionary (for benchmarks).
    """
    chain = get_sqp_chain()
    history = create_chat_history(messages)
    
    try:
        # The chain returns a dictionary with all the steps
        result_dict = chain.invoke({
            "question": question,
            "input": question,
            "messages": history.messages,
        })
        
        # Format the final answer string for the UI
        final_answer_str = format_answer(
            question=result_dict["question"], 
            query=result_dict["cleaned_query"], 
            result=result_dict["result"],
            table_info=result_dict["filtered_schema"]
        )
        
        # Add the final string to the dictionary for the benchmark script
        result_dict["final_answer_string"] = final_answer_str

    except IndexError:
        # This specific error happens when the SILE fails to return a query plan.
        print(f"SILE returned an empty plan for query: '{question}'")
        error_str = "I'm sorry, but I could not devise a query plan for your question."
        if return_dict:
            return {
                "final_answer_string": error_str,
                "generated_sql": "SILE_FAILURE",
                "execution_result": "ERROR: SILE returned an empty plan."
            }
        return error_str

    except Exception as e:
        print(f"ERROR in SQP chain invocation: {e}")
        error_str = "I'm sorry, but I encountered an error processing your SQL query."
        if return_dict:
            return {
                "final_answer_string": error_str,
                "generated_sql": "ERROR",
                "execution_result": f"ERROR: {e}"
            }
        return error_str
        
    if return_dict:
        return result_dict
    else:
        return result_dict["final_answer_string"]