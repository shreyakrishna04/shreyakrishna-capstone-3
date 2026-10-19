
"""Validated, self-contained SQL agent with the shared tool interface."""

import os
import re

import psycopg2
from dotenv import load_dotenv

from bedrock_client import invoke_claude


load_dotenv()


SCHEMA_CONTEXT = """
Tables available:

- clients(id, name, industry, region, contract_value, created_date)
- projects(id, client_id, name, status, start_date, end_date, budget)
- consultants(id, name, grade, specialisation, project_id, billable_hours)
- invoices(id, project_id, amount, status, due_date, paid_date)

All monetary values are in USD. Dates are in ISO 8601 format.
projects.status is one of: 'active', 'completed', 'on-hold'.
"""


BLOCKED_KEYWORDS = (
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "TRUNCATE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
)

UNSAFE_INTENT_PATTERN = re.compile(
    r"\b("
    r"delete|drop|update|insert|alter|truncate|"
    r"modify|change|remove|erase|destroy|purge"
    r")\b",
    re.IGNORECASE,
)


def contains_unsafe_intent(question: str) -> bool:
    return bool(UNSAFE_INTENT_PATTERN.search(question or ""))

def _token_usage(result: dict | None) -> dict:
    result = result or {}
    return {
        "input": int(result.get("input_tokens", 0)),
        "output": int(result.get("output_tokens", 0)),
    }


def _add_token_usage(*usages: dict) -> dict:
    return {
        "input": sum(int(usage.get("input", 0)) for usage in usages),
        "output": sum(int(usage.get("output", 0)) for usage in usages),
    }


def _clean_generated_sql(text: str) -> str:
    sql = text.strip()

    if sql.startswith("```sql"):
        sql = sql[len("```sql"):].strip()
    elif sql.startswith("```"):
        sql = sql[len("```"):].strip()

    if sql.endswith("```"):
        sql = sql[:-3].strip()

    return sql


def validate_sql(query: str) -> dict:
    """
    Validate SQL before any database connection or execution.

    Rules:
    1. Only SELECT statements are allowed.
    2. Block destructive keywords as whole words.
    3. Block stacked statements independently, including SELECT;SELECT.
    """
    if not isinstance(query, str) or not query.strip():
        return {"valid": False, "reason": "Query is empty."}

    normalized = query.strip()
    upper_query = normalized.upper()

    for keyword in BLOCKED_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper_query):
            return {
                "valid": False,
                "reason": f"Blocked: '{keyword}' is not permitted.",
            }

    if not re.match(r"^SELECT\b", upper_query):
        return {
            "valid": False,
            "reason": "Only SELECT queries are permitted.",
        }

    query_without_one_trailing_semicolon = normalized
    if query_without_one_trailing_semicolon.endswith(";"):
        query_without_one_trailing_semicolon = (
            query_without_one_trailing_semicolon[:-1].rstrip()
        )

    if ";" in query_without_one_trailing_semicolon:
        return {
            "valid": False,
            "reason": "Blocked: multiple statements detected.",
        }

    return {"valid": True, "reason": "OK"}


def generate_sql_with_usage(question: str) -> tuple[str, dict]:
    prompt = (
        f"{SCHEMA_CONTEXT}\n\n"
        f"Generate one read-only SQL query to answer: {question}\n\n"
        "Return ONLY one SQL SELECT query. "
        "Do not return explanations, markdown, or backticks."
        "Use only SELECT statements. "
        "For active clients, join clients.id to projects.client_id "
        "and filter projects.status = 'active'. "
        "Never generate DELETE, UPDATE, INSERT, DROP, ALTER, or TRUNCATE. "
    )

    result = invoke_claude(
        prompt,
        max_tokens=512,
        component="sql_generation",
    )

    return _clean_generated_sql(result["text"]), _token_usage(result)


def generate_sql(question: str) -> str:
    """Generate SQL while preserving the original public helper interface."""
    sql, _ = generate_sql_with_usage(question)
    return sql


def execute_query_safely(sql: str) -> dict:
    """
    Validate first, then connect and execute only validated SQL.

    This is the only function in this module that accesses RDS.
    """
    validation = validate_sql(sql)

    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["reason"],
            "validation": "BLOCKED",
            "sql": sql,
        }

    connection = None
    cursor = None

    try:
        connection = psycopg2.connect(
            host=os.getenv("RDS_HOST"),
            port=int(os.getenv("RDS_PORT", "5432")),
            dbname=os.getenv("RDS_DATABASE"),
            user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASSWORD"),
        )
        connection.set_session(readonly=True)
        cursor = connection.cursor()
        cursor.execute(sql)

        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]

        return {
            "success": True,
            "columns": columns,
            "rows": rows,
            "validation": "PASSED",
            "sql": sql,
        }
    except Exception as error:
        return {
            "success": False,
            "error": str(error),
            "validation": "EXECUTION_ERROR",
            "sql": sql,
        }
    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def validated_sql_agent(question: str) -> dict:
    
    if contains_unsafe_intent(question):
        return {
            "success": False,
            "answer": (
                "Blocked: this request asks to modify or delete database data. "
                "Only read-only SELECT queries are permitted."
            ),
            "sql": "",
            "validation": "BLOCKED",
            "token_usage": {"input": 0, "output": 0},
        }

    sql, generation_usage = generate_sql_with_usage(question)
    execution = execute_query_safely(sql)
    
    sql, generation_usage = generate_sql_with_usage(question)
    execution = execute_query_safely(sql)

    if not execution["success"]:
        return {
            "success": False,
            "answer": f"Query blocked or failed: {execution['error']}",
            "sql": sql,
            "validation": execution["validation"],
            "token_usage": generation_usage,
        }

    interpretation_prompt = (
        f"Question: {question}\n"
        f"Columns: {execution['columns']}\n"
        f"Data: {execution['rows'][:20]}\n\n"
        "Answer concisely using only the returned data."
    )

    try:
        interpretation = invoke_claude(
            interpretation_prompt,
            max_tokens=500,
            component="sql_interpretation",
        )
    except Exception as error:
        return {
            "success": False,
            "answer": f"Query executed, but interpretation failed: {error}",
            "sql": sql,
            "validation": "INTERPRETATION_ERROR",
            "token_usage": generation_usage,
        }

    return {
        "success": True,
        "answer": interpretation["text"],
        "sql": sql,
        "validation": "PASSED",
        "token_usage": _add_token_usage(
            generation_usage,
            _token_usage(interpretation),
        ),
    }


def run_sql_agent(question: str) -> dict:
    result = validated_sql_agent(question)

    sources = []
    if result["sql"]:
        sources = [f"SQL: {result['sql']}"]

    return {
        "success": result["success"],
        "answer": result["answer"],
        "sources": sources,
        "metadata": {
            "validation": result["validation"],
            "token_usage": result["token_usage"],
        },
    }
