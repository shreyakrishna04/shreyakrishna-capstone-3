
"""Tracing, routing evidence, and tokenomics summaries."""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from bedrock_client import (
    TOKENOMICS_LOG_PATH,
    reset_trace_id,
    set_trace_id,
)
from orchestrator import route_question


DECISION_LOG_PATH = Path(
    os.getenv("DECISION_LOG_PATH", "/tmp/agent_decisions.jsonl")
)


def log_decision(stage: str, question: str, details: dict) -> None:
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "question": question,
        "details": details,
    }

    print(f"[LOG] {stage}: {json.dumps(details, default=str)}")
    DECISION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DECISION_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(entry, default=str) + "\n")


def route_question_logged(state):
    result = route_question(state)
    log_decision("routing", state["question"], {"route": result["route"]})
    return result


def summarize_tokenomics(trace_id: str | None = None) -> dict:
    by_component = defaultdict(
        lambda: {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    )

    if not TOKENOMICS_LOG_PATH.exists():
        return {
            "records": 0,
            "by_component": {},
            "total_cost_usd": 0.0,
            "most_tokens_component": None,
        }

    with TOKENOMICS_LOG_PATH.open(encoding="utf-8") as log_file:
        for line in log_file:
            if not line.strip():
                continue

            record = json.loads(line)

            if trace_id is not None and record.get("trace_id") != trace_id:
                continue

            component = record.get("component", "unspecified")
            summary = by_component[component]
            summary["calls"] += 1
            summary["input_tokens"] += int(record.get("input_tokens", 0))
            summary["output_tokens"] += int(record.get("output_tokens", 0))
            summary["total_tokens"] += int(record.get("total_tokens", 0))
            summary["cost_usd"] += float(record.get("cost_usd", 0.0))

    component_summary = {
        component: {
            **values,
            "cost_usd": round(values["cost_usd"], 10),
        }
        for component, values in by_component.items()
    }

    total_cost = sum(
        values["cost_usd"] for values in component_summary.values()
    )

    most_tokens_component = None
    if component_summary:
        most_tokens_component = max(
            component_summary,
            key=lambda name: component_summary[name]["total_tokens"],
        )

    return {
        "records": sum(
            values["calls"] for values in component_summary.values()
        ),
        "by_component": component_summary,
        "total_cost_usd": round(total_cost, 10),
        "most_tokens_component": most_tokens_component,
    }


def print_tokenomics_summary() -> dict:
    summary = summarize_tokenomics()

    print("\nTOKENOMICS SUMMARY")
    print("-" * 80)

    for component, values in summary["by_component"].items():
        print(
            f"{component}: "
            f"calls={values['calls']} "
            f"input={values['input_tokens']} "
            f"output={values['output_tokens']} "
            f"total={values['total_tokens']} "
            f"cost_usd={values['cost_usd']:.10f}"
        )

    print("-" * 80)
    print(f"Total cost: ${summary['total_cost_usd']:.10f}")
    print(
        "Most tokens: "
        f"{summary['most_tokens_component'] or 'none'}"
    )
    return summary


def run_traced_query(orchestrator, question: str) -> dict:
    trace_id = str(uuid.uuid4())
    trace_token = set_trace_id(trace_id)

    log_decision("query_start", question, {"trace_id": trace_id})

    try:
        result = orchestrator.invoke(
            {
                "question": question,
                "route": None,
                "sql_result": None,
                "retrieval_result": None,
                "final_answer": None,
                "token_usage": None,
            }
        )

        query_tokenomics = summarize_tokenomics(trace_id)

        log_decision(
            "query_complete",
            question,
            {
                "trace_id": trace_id,
                "route": result["route"],
                "sql_validation": (
                    result.get("sql_result", {})
                    .get("metadata", {})
                    .get("validation")
                    if result.get("sql_result")
                    else None
                ),
                "retrieval_source_count": (
                    len(result.get("retrieval_result", {}).get("sources", []))
                    if result.get("retrieval_result")
                    else None
                ),
                "token_usage": result.get("token_usage"),
                "tokenomics": query_tokenomics,
            },
        )

        return result
    finally:
        reset_trace_id(trace_token)
