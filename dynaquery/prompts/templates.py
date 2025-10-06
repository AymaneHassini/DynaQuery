# prompts/template.py
"""
Centralized prompt templates for LLM interactions.
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    FewShotPromptTemplate
)

# Answer Prompt Template
answer_prompt = PromptTemplate.from_template(
"""Given the following user question, corresponding SQL query, and SQL result, answer the user question.
Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: """
)

# --- EXPLICIT JOIN PROMPT ---
join_prompt = ChatPromptTemplate.from_template(
"""You are a SQL expert. You are given a database schema, an explicit base table, and a list of tables to join.
Your task is to generate the appropriate SQL LEFT JOIN clauses to join the base table to each of the other tables.
Use the foreign key relationships defined in the schema to determine the ON conditions.

Schema:
{schema}

Base Table:
{base_table}

Tables to Join:
{join_tables}

If no valid join condition can be inferred from the schema for a given table, do not include it.
If no join conditions can be inferred at all, output exactly "No join".
Output only the JOIN clauses themselves, without the initial SELECT statement.
"""
)

# Per-row Reasoning Guidelines
REASONING_GUIDELINES = """
REASONING RULES:
1. You will see every field in this record—treat them all as evidence.
2. Walk through each field step by step (chain-of-thought) to decide if this record matches the user's question.
3. Use your reasoning and inference/context understanding capabilities.
"""

# Final SQL Generation Prompt
def create_zero_shot_prompt():
    """Create the final zero-shot prompt template for SQL generation."""
    return ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a MySQL expert. Given a user question and a database schema, create a syntactically correct MySQL query.

Use this schema:
{table_info}

GUIDELINES:
1. Pay close attention to the user's question to determine the exact columns and ordering required.
2. Do NOT add a LIMIT clause unless the user explicitly asks for a specific number of results.
3. Use explicit JOINs where necessary based on the foreign key relationships.
4. Use table aliases for clarity in JOINs.
5. Ensure the generated query is only a single SQL statement.
6. [CRITICAL]: If you cannot generate a query to answer the question based on the provided schema, you MUST return the following SQL query exactly: `SELECT 'error'`
"""
        ),
        MessagesPlaceholder(variable_name="messages"), 
        ("human", "{input}"),
    ])
def create_few_shot_prompt():
    """
    Creates a dynamic few-shot prompt template for SQL generation,
    designed to work with a SemanticSimilarityExampleSelector.
    """
    # 1. Define the format for each individual few-shot example.
    example_prompt = PromptTemplate.from_template(
        "User Question: {input}\nSQL Query: {query}"
    )
    
    # 2. Define the overall prompt structure that will incorporate the examples.
    few_shot_prompt = FewShotPromptTemplate(
        example_selector=None, 
        example_prompt=example_prompt,
        prefix="""You are a MySQL expert. Given a user question and a database schema, create a syntactically correct MySQL query.
You are given some examples of user questions and their corresponding SQL queries.

Use this schema:
{table_info}""",
        # The text that comes after the examples.
        suffix="User Question: {input}\nSQL Query:",
        input_variables=["input", "table_info"],
    )
    
    return ChatPromptTemplate.from_messages([
        ("system", few_shot_prompt.prefix),
        MessagesPlaceholder(variable_name="messages"),
        ("human", few_shot_prompt.suffix),
    ])
