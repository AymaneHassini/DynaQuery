# main.py
"""
Streamlit application for the DynaQuery Framework.
Provides a user interface to interact with and compare the Structured Query Pipeline (SQP)
and the Generalized Multimodal Pipeline (MMP).
"""
import streamlit as st
from chains.sqp import invoke_sqp
from chains.mmp import invoke_mmp

def main():
    """Main application entry point."""
    st.title("DynaQuery: Live Demo")
    
    # --- Initialize Session State ---
    # We only need three state variables
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = ""
    if "last_pipeline_used" not in st.session_state:
        st.session_state.last_pipeline_used = ""

    # --- UI Elements ---
    # Mode selection for the *next* query submitted via the chat input
    mode = st.radio(
        "Default Pipeline for New Queries", 
        ["Structured Query Pipeline (SQP)", "Generalized Multimodal Pipeline (MMP)"]
    )
    
    # Display the entire conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # --- Handle New User Input from Chat Box ---
    if user_query := st.chat_input("Enter your query..."):
        # Update state with the latest query and the pipeline we are about to use
        st.session_state.last_user_query = user_query
        st.session_state.last_pipeline_used = mode
        
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner(f"Processing with {mode}..."):
                if "SQP" in mode:
                    response = invoke_sqp(user_query, st.session_state.messages)
                else: # MMP mode
                    response = invoke_mmp(user_query, st.session_state.messages)
                st.markdown(response)
        
        # Add assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        # Rerun to ensure the "re-run" button appears immediately
        st.rerun()

    # --- "Try the Other Pipeline" Button Logic ---
    # This button only appears after at least one query has been run
    if st.session_state.last_user_query:
        # Determine which pipeline is the "other" one
        if "SQP" in st.session_state.last_pipeline_used:
            other_pipeline_name = "Generalized Multimodal Pipeline (MMP)"
            button_label = f"🔍 Re-run with {other_pipeline_name}"
        else:
            other_pipeline_name = "Structured Query Pipeline (SQP)"
            button_label = f"🔍 Re-run with {other_pipeline_name}"

        if st.button(button_label):
            query_to_rerun = st.session_state.last_user_query
            
            # Update the state to reflect the new pipeline being used
            st.session_state.last_pipeline_used = other_pipeline_name
            
            # Add a message to the chat to inform the user what's happening
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Running the last query ('{query_to_rerun}') again using the **{other_pipeline_name}**..."
            })

            # Generate and display the new response
            with st.chat_message("assistant"):
                with st.spinner(f"Processing with {other_pipeline_name}..."):
                    if "SQP" in other_pipeline_name:
                        response = invoke_sqp(query_to_rerun, st.session_state.messages)
                    else: # MMP mode
                        response = invoke_mmp(query_to_rerun, st.session_state.messages)
                    st.markdown(response)
            
            # Add the new response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            # Rerun the script to update the display instantly
            st.rerun()

if __name__ == "__main__":
    main()