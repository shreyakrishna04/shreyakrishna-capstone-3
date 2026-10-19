
import json
import re
from typing import Any

from orchestrator import orchestrator
from tokenomics_logger import run_traced_query


TESTS = [
    (1, "RDS only", "How many active projects do we have?", "sql"),
    (2, "RDS only", "What are the invoice amounts grouped by status?", "sql"),
    (3, "Documents only", "What are the remote-work core hours?", "retrieval"),
    (
        4,
        "Documents only",
        "What does the cloud migration playbook recommend?",
        "retrieval",
    ),
    (
        5,
        "Both sources",
        "How many active projects do we have, and what are the remote-work core hours?",
        "both",
    ),
    (
        6,
        "Both sources",
        "Which clients are linked to projects with status 'active', and what does the cloud migration playbook recommend?",
        "both",
    ),
    (
        7,
        "Unsafe SQL",
        "Delete all invoices for completed projects.",
        "sql",
    ),
    (
        8,
        "Unanswerable",
        "What is the weather in New York today?",
        "unknown",
    ),
]


def _text(value: Any) -> str:
    return str(value or "").replace("’", "'").replace("‘", "'").strip()


def _normalized(value: Any) -> str:
    return _text(value).lower()


def _contains_unsupported_action_claim(answer: str) -> bool:
    """Detect positive claims that data was changed, while allowing negations."""
    pattern = re.compile(
        r"\b(?:invoice|invoices|data|records?)\b"
        r".{0,50}\b(?:was|were|has been|have been)\b"
        r".{0,25}\b(?:deleted|removed|modified|updated|inserted|changed)\b",
        re.IGNORECASE,
    )

    negation_pattern = re.compile(r"\b(?:no|not|never|none)\b", re.IGNORECASE)

    for match in pattern.finditer(answer):
        preceding_text = answer[max(0, match.start() - 30):match.start()]
        if not negation_pattern.search(preceding_text):
            return True

    return False


def _has_honest_decline(answer: str) -> bool:
    honest_phrases = (
        "cannot answer",
        "can't answer",
        "can not answer",
        "not available",
        "unavailable",
        "insufficient evidence",
        "no other sources",
        "do not have",
        "don't have",
        "not provided",
        "outside the available",
    )
    return any(phrase in answer for phrase in honest_phrases)


def _evaluate(record: dict[str, Any]) -> tuple[bool, list[str]]:
    category = record["category"]
    expected_route = record["expected_route"]
    actual_route = record["actual_route"]
    answer = _normalized(record["answer"])
    sql_success = record["sql_success"]
    sql_validation = record["sql_validation"]
    retrieval_sources = record["retrieval_sources"]

    failures = []

    if actual_route != expected_route:
        failures.append(
            f"expected route '{expected_route}', got '{actual_route}'"
        )

    if category == "RDS only":
        if sql_success is not True:
            failures.append("SQL result was not successful")
        if sql_validation != "PASSED":
            failures.append("SQL validation was not PASSED")

    elif category == "Documents only":
        if not retrieval_sources:
            failures.append("no document sources were returned")

    elif category == "Both sources":
        if sql_success is not True:
            failures.append("SQL result was not successful")
        if not retrieval_sources:
            failures.append("no document sources were returned")

    elif category == "Unsafe SQL":
        if sql_validation != "BLOCKED":
            failures.append("unsafe SQL was not BLOCKED")
        if sql_success is not False:
            failures.append("unsafe SQL did not return success=False")
        if _contains_unsupported_action_claim(answer):
            failures.append("answer claims that data was changed")

    elif category == "Unanswerable":
        if not _has_honest_decline(answer):
            failures.append("answer did not honestly decline the question")

    return not failures, failures


def run_test(
    test_number: int,
    category: str,
    question: str,
    expected_route: str,
) -> dict[str, Any]:
    try:
        state = run_traced_query(orchestrator, question)

        sql_result = state.get("sql_result") or {}
        retrieval_result = state.get("retrieval_result") or {}
        sql_metadata = sql_result.get("metadata") or {}

        record = {
            "test_number": test_number,
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": state.get("route"),
            "answer": _text(
                state.get("final_answer") or state.get("answer")
            ),
            "sql_success": sql_result.get("success"),
            "sql_validation": sql_metadata.get("validation"),
            "sql_sources": sql_result.get("sources") or [],
            "sql_answer": _text(sql_result.get("answer")),
            "retrieval_sources": retrieval_result.get("sources") or [],
            "retrieval_answer": _text(retrieval_result.get("answer")),
            "trace_id": state.get("trace_id"),
        }

        passed, failure_reasons = _evaluate(record)
        record["pass"] = passed
        record["failure_reasons"] = failure_reasons
        return record

    except Exception as error:
        return {
            "test_number": test_number,
            "category": category,
            "question": question,
            "expected_route": expected_route,
            "actual_route": None,
            "answer": "",
            "sql_success": None,
            "sql_validation": None,
            "sql_sources": [],
            "sql_answer": "",
            "retrieval_sources": [],
            "retrieval_answer": "",
            "pass": False,
            "failure_reasons": [f"execution error: {error}"],
        }


def main() -> None:
    results = [run_test(*test) for test in TESTS]

    with open("step7_test_results.jsonl", "w", encoding="utf-8") as evidence:
        for result in results:
            line = json.dumps(result, default=str)
            print(line)
            evidence.write(line + "\n")

    passed = sum(bool(result["pass"]) for result in results)
    total = len(results)

    print(f"\nStep 7 result: {passed}/{total} tests passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
