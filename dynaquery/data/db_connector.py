# data/db_connector.py

"""
Database connection utilities for the application.
Provides connection objects and methods for database access.
"""
import mysql.connector
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine import Engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from dotenv import load_dotenv, find_dotenv

from config.settings import DB_HOST, DB_USER, DB_PASSWORD, DB_NAME, get_mysql_uri

def get_mysql_connection():
    """
    Create and return a MySQL connection object.
    """
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

def get_sqlalchemy_engine() -> Engine:
    """
    Create and return an SQLAlchemy engine.
    """
    return create_engine(get_mysql_uri())

def get_inspector():
    """
    Return an SQLAlchemy inspector object.
    """
    engine = get_sqlalchemy_engine()
    return inspect(engine)

def get_langchain_db() -> SQLDatabase:
    """
    Create and return a LangChain SQLDatabase object.
    """
    return SQLDatabase.from_uri(get_mysql_uri())

def get_query_tool() -> QuerySQLDataBaseTool:
    """
    Create and return a LangChain query tool.
    """
    db = get_langchain_db()
    return QuerySQLDataBaseTool(db=db)