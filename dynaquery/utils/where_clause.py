from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dynaquery.models.llm import get_langchain_llm

# 1. Define the prompt template
# Make sure to change MySQL to the dialect of sql you are using. 
WHERE_PROMPT_TEMPLATE = """You are a MySQL expert. Your task is to analyze the user's question and generate the filter conditions for a SQL WHERE clause.

You have been given a query plan that has already identified the necessary tables. You MUST use only the tables listed in the query plan.

**Query Plan:**
- Base Table: {base_table}
- Join Tables: {join_tables}

**Schema for these tables:**
{schema}

**User's Question:**
{input}

Based on the user's question and the provided query plan, extract ONLY the filter conditions.
- Do NOT include the 'WHERE' keyword.
- Combine multiple conditions with 'AND'.
- If there are no filter conditions, return the exact string "NO_CONDITIONS".
- IMPORTANT: Use full table names (e.g., `products.column`), not aliases.
- **[CRITICAL RULE] Be very careful. Only extract conditions that refer to the structured data in the schema (IDs, names, dates, numbers, cities, etc.). Do NOT try to translate visual or physical descriptions (like colors, shapes, materials...*) into SQL conditions. These visual descriptions will be handled by a different system.**

**Extracted Conditions:**"""

where_prompt = ChatPromptTemplate.from_template(WHERE_PROMPT_TEMPLATE)

# 2. Create the chain
def create_where_clause_chain():
    """Creates a LangChain for generating SQL WHERE clauses."""
    llm = get_langchain_llm()
    return where_prompt | llm | StrOutputParser()