"""
tokenomics_logger.py -- structured logging and full-chain tracing for the
orchestrator, kept separate from orchestrator.py itself, the same way
bedrock_client.py stays separate from every agent that calls it.
"""
import json
import uuid
from datetime import datetime
from orchestrator import route_question


def log_decision(stage: str, question: str, details: dict):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "question": question,
        "details": details
    }
    print(f"[LOG] {stage}: {json.dumps(details)}")
    with open("/tmp/agent_decisions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def route_question_logged(state):
    result = route_question(state)
    log_decision("routing", state["question"], {"route": result["route"]})
    return result


def run_traced_query(orchestrator, question: str) -> dict:
    trace_id = str(uuid.uuid4())[:8]

    log_decision("query_start", question, {"trace_id": trace_id})

    result = orchestrator.invoke({
        "question": question, "route": None, "sql_result": None,
        "retrieval_result": None, "final_answer": None, "token_usage": None
    })

    # NOTE: sql_result's "validation" field is nested under "metadata", not
    # top-level -- run_sql_agent() returns {success, answer, sources,
    # metadata: {validation, token_usage}}. Accessing result["sql_result"]
    # ["validation"] directly silently returns None even on a query that
    # actually passed. This was a real, confirmed bug in an earlier version
    # of this file -- verified directly against the actual return shape.
    log_decision("query_complete", question, {
        "trace_id": trace_id,
        "route": result["route"],
        "sql_validation": result.get("sql_result", {}).get("metadata", {}).get("validation") if result.get("sql_result") else None,
        "retrieval_chunk_count": len(result.get("retrieval_result", {}).get("sources", [])) if result.get("retrieval_result") else None,
        "token_usage": result.get("token_usage")
    })

    return result
