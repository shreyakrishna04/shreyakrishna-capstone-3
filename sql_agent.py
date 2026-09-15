"""
agents/sql_agent.py -- validated SQL Query Agent, wrapped to the standard
{success, answer, sources, metadata} interface from the Shared Tool Registry
lab.

NOTE ON STRUCTURE: earlier in this program, run_sql_agent() reached into a
sibling repo via sys.path.append("../add-validation-layer-lab") to import
validated_sql_agent. That works for local development, where both repos'
directories physically exist side by side -- it does NOT work in a deployed
Lambda package, which only has access to what's actually bundled inside the
zip. Every piece of logic (generate_sql, validate_sql, execute_query_safely)
is inlined directly here instead, so this file is genuinely self-contained
and deployable on its own.
"""
import os
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
status in projects is one of: 'active', 'completed', 'on-hold'.
"""

BLOCKED_KEYWORDS = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE",
                    "EXEC", "EXECUTE", "GRANT", "REVOKE"]


def validate_sql(query: str) -> dict:
    """
    Blocks any query that isn't a SELECT, contains a destructive keyword, or
    stacks multiple statements. This must run before any database call.
    """
    query_upper = query.upper().strip()

    for keyword in BLOCKED_KEYWORDS:
        if f" {keyword} " in f" {query_upper} " or query_upper.startswith(keyword):
            return {"valid": False, "reason": f"Blocked: '{keyword}' is not permitted."}

    if not query_upper.startswith("SELECT"):
        return {"valid": False, "reason": "Only SELECT queries are permitted."}

    # Block stacked queries independently of keyword content -- a stacked
    # query with two harmless-looking SELECTs would sail past every keyword
    # check above, since there's nothing destructive in either half.
    stripped = query.strip().rstrip(";")
    if ";" in stripped:
        return {"valid": False, "reason": "Blocked: multiple statements detected."}

    return {"valid": True, "reason": "OK"}


def generate_sql(question: str) -> str:
    prompt = (
        f"{SCHEMA_CONTEXT}\n\n"
        f"Generate a SQL query to answer: {question}\n\n"
        "Return ONLY the SQL query. No explanation, no markdown, no backticks."
    )
    result = invoke_claude(prompt, max_tokens=512)
    sql = result["text"].strip()
    return sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


def execute_query_safely(sql: str) -> dict:
    validation = validate_sql(sql)
    if not validation["valid"]:
        return {"success": False, "error": validation["reason"], "validation": "BLOCKED", "sql": sql}

    try:
        conn = psycopg2.connect(
            host=os.getenv("RDS_HOST"), port=int(os.getenv("RDS_PORT")),
            dbname=os.getenv("RDS_DATABASE"), user=os.getenv("RDS_USER"),
            password=os.getenv("RDS_PASSWORD")
        )
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        conn.close()
        return {"success": True, "columns": cols, "rows": rows, "validation": "PASSED", "sql": sql}
    except Exception as e:
        return {"success": False, "error": str(e), "validation": "EXECUTION_ERROR", "sql": sql}


def validated_sql_agent(question: str) -> dict:
    sql = generate_sql(question)
    result = execute_query_safely(sql)

    if not result["success"]:
        return {
            "answer": f"Query blocked or failed: {result['error']}",
            "sql": sql,
            "validation": result["validation"],
            "token_usage": {"input": 0, "output": 0},
        }

    prompt = (
        f"Question: {question}\nColumns: {result['columns']}\n"
        f"Data: {result['rows'][:20]}\n\nAnswer concisely."
    )
    interpretation = invoke_claude(prompt, max_tokens=500)

    return {
        "answer": interpretation["text"],
        "sql": sql,
        "validation": result["validation"],
        "token_usage": {"input": interpretation["input_tokens"], "output": interpretation["output_tokens"]},
    }


def run_sql_agent(question: str) -> dict:
    """Standardized wrapper matching the Shared Tool Registry lab's interface."""
    result = validated_sql_agent(question)
    return {
        "success": result["validation"] == "PASSED",
        "answer": result["answer"],
        "sources": [f"SQL: {result['sql']}"],
        "metadata": {"validation": result["validation"], "token_usage": result["token_usage"]},
    }
