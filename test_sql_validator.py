
from agents.sql_agent import validate_sql


TEST_CASES = [
    (
        "Valid SELECT",
        "SELECT COUNT(*) FROM projects;",
        True,
    ),
    (
        "Destructive DELETE",
        "DELETE FROM projects WHERE id = 1;",
        False,
    ),
    (
        "Column containing blocked word",
        "SELECT deleted_at FROM clients;",
        True,
    ),
    (
        "Stacked query containing blocked keyword",
        "SELECT id FROM projects; DELETE FROM projects;",
        False,
    ),
    (
        "Harmless stacked SELECT query",
        "SELECT id FROM projects; SELECT id FROM clients;",
        False,
    ),
]


def main():
    failures = []

    print("SQL VALIDATOR TEST RESULTS")

    for name, query, expected_valid in TEST_CASES:
        result = validate_sql(query)
        actual_valid = result["valid"]
        passed = actual_valid == expected_valid

        print(f"\nTest: {name}")
        print(f"Query: {query}")
        print(f"Expected valid: {expected_valid}")
        print(f"Actual valid: {actual_valid}")
        print(f"Reason: {result['reason']}")
        print(f"Status: {'PASS' if passed else 'FAIL'}")

        if not passed:
            failures.append(name)

    print("\nOVERALL RESULT")

    if failures:
        print(f"FAIL: {', '.join(failures)}")
        raise SystemExit(1)

    print("PASS: All five SQL validator cases passed.")


if __name__ == "__main__":
    main()
