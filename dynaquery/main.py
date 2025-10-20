# main.py
"""
Streamlit application for the DynaQuery Framework.
Provides a user interface to interact with and compare SQP and the two versions of MMP,
preserving the original simple layout with enhanced comparison and UI locking.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from dynaquery.chains.sqp import invoke_sqp
from dynaquery.chains.mmp import invoke_mmp

def main():
    """Main application entry point."""
    st.title("DynaQuery: Live Demo")
    
    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = ""
    if "last_pipeline_used" not in st.session_state:
        st.session_state.last_pipeline_used = ""
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # --- UI Elements ---
    # --- define the list of pipelines to match the radio button options ---
    ALL_PIPELINES = [
        "Structured Query Pipeline (SQP)", 
        "MMP (BERT Classifier)", 
        "MMP (LLM-Native Classifier)"
    ]
    
    mode = st.radio(
        "Default Pipeline for New Queries", 
        options=ALL_PIPELINES, # use the defined list
        disabled=st.session_state.processing
    )
    
    # display the entire conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # --- handle new user input from chat box ---
    if user_query := st.chat_input("Enter your query...", disabled=st.session_state.processing):
        st.session_state.processing = True
        st.session_state.last_user_query = user_query
        st.session_state.last_pipeline_used = mode
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()

    # --- central processing block ---
    if st.session_state.processing:
        query_to_run = st.session_state.last_user_query
        pipeline_to_run = st.session_state.last_pipeline_used

        info_message = f"Running query with the **{pipeline_to_run}**..."
        st.session_state.messages.append({"role": "assistant", "content": info_message})
        with st.chat_message("assistant"):
            st.markdown(info_message)

        with st.chat_message("assistant"):
            with st.spinner(f"Processing with {pipeline_to_run}..."):
                if "SQP" in pipeline_to_run:
                    response_obj = invoke_sqp(query_to_run, st.session_state.messages, return_dict=True)
                    response = response_obj.get("final_answer_string", "An error occurred in SQP.")

                elif "BERT" in pipeline_to_run:
                    response_obj = invoke_mmp(query_to_run, st.session_state.messages, classifier_type="bert")
                    response = response_obj.get("final_answer_string", "An error occurred in MMP-BERT.")
                else: # LLM-Native
                    response_obj = invoke_mmp(query_to_run, st.session_state.messages, classifier_type="llm")
                    response = response_obj.get("final_answer_string", "An error occurred in MMP-LLM.")
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.processing = False
        st.rerun()

    if st.session_state.last_user_query and not st.session_state.processing:
        last_run = st.session_state.last_pipeline_used
        other_pipelines = [p for p in ALL_PIPELINES if p != last_run]
        st.write("---")
        col1, col2 = st.columns(2)

        if col1.button(f"🔍 Re-run with {other_pipelines[0]}"):
            st.session_state.last_pipeline_used = other_pipelines[0]
            st.session_state.processing = True
            st.rerun()

        if col2.button(f"🔍 Re-run with {other_pipelines[1]}"):
            st.session_state.last_pipeline_used = other_pipelines[1]
            st.session_state.processing = True
            st.rerun()

if __name__ == "__main__":
    main()