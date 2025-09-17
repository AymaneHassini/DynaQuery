# models/llm.py
"""
LLM initialization and configuration.
"""
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GOOGLE_API_KEY, LLM_MODEL, TEMPERATURE

def load_llm():
    """
    Load and configure the Gemini generative model.
    
    Returns:
        Configured Gemini generative model
    """
    # Configure the Gemini API
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Return the model
    return genai.GenerativeModel(LLM_MODEL)

def get_langchain_llm():
    """
    Get a LangChain-compatible LLM.
    
    Returns:
        LangChain ChatGoogleGenerativeAI instance
    """
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE)