from agents.sql_agent import run_sql_agent
from agents.retrieval_agent import run_retrieval_agent


EXPECTED_KEYS = {"success", "answer", "sources", "metadata"}


def check_tool(name, function, question):
    result = function(question)
    actual_keys = set(result.keys())

    print(f"\n{name}")
    print(f"Question: {question}")
    print(f"Top-level keys: {sorted(actual_keys)}")
    print(f"Metadata: {result.get('metadata')}")
    print(f"Sources: {result.get('sources')}")

    if actual_keys != EXPECTED_KEYS:
        raise AssertionError(
            f"{name} returned {sorted(actual_keys)}, "
            f"expected {sorted(EXPECTED_KEYS)}"
        )

    return result


def main():
    check_tool(
        "SQL agent",
        run_sql_agent,
        "How many active projects do we have?",
    )

    check_tool(
        "Retrieval agent",
        run_retrieval_agent,
        "Summarize the remote work policy.",
    )

    print("\nPASS: Both tools return the exact shared interface.")


if __name__ == "__main__":
    main()
