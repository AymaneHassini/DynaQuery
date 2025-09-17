# config/settings.py

"""
Centralized configuration management for the application.
Loads environment variables and provides access to them.
"""
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Database configuration
DB_HOST = os.getenv("db_host")
DB_USER = os.getenv("db_user")
DB_PASSWORD = os.getenv("db_password")
DB_NAME = os.getenv("db_name")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")

# Model settings
LLM_MODEL = "gemini-2.5-pro"
TEMPERATURE = 0
CHECKPOINT_PATH = "/Users/aymenhassini/Desktop/working-copy/checkpoint"  # Change to the Path to BERT model checkpoint

# Database URI builder
def get_mysql_uri():
    """Build a MySQL connection URI using environment variables."""
    return f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"

# Validation
def validate_config():
    """Validate that all required environment variables are present."""
    required_vars = [DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, GOOGLE_API_KEY]
    if not all(required_vars):
        missing = [name for name, value in {
            'DB_HOST': DB_HOST, 
            'DB_USER': DB_USER, 
            'DB_PASSWORD': DB_PASSWORD, 
            'DB_NAME': DB_NAME,
            'GOOGLE_API_KEY': GOOGLE_API_KEY
        }.items() if not value]
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return True

# Initialize validation on import
validate_config()