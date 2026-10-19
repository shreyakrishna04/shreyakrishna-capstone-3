
from agents.retrieval_agent import get_opensearch_client
from agents.sql_agent import run_sql_agent


TABLES = ("clients", "projects", "consultants", "invoices")


def main():
    failures = []

    print("RDS FOUNDATION CHECK")
    for table in TABLES:
        question = (
            f"Return only the row count for the {table} table. "
            f"Use exactly: SELECT COUNT(*) AS row_count FROM {table};"
        )
        result = run_sql_agent(question)

        print(f"\nTable: {table}")
        print(f"Success: {result.get('success')}")
        print(f"Answer: {result.get('answer')}")
        print(f"Sources: {result.get('sources')}")
        print(f"Metadata: {result.get('metadata')}")

        if not result.get("success"):
            failures.append(table)

    print("\nOPENSEARCH FOUNDATION CHECK")
    try:
        client = get_opensearch_client()
        response = client.count(
            index="documents",
            body={"query": {"match_all": {}}},
        )
        document_count = response["count"]
        print(f"Index: documents")
        print(f"Document count: {document_count}")

        if document_count <= 0:
            failures.append("OpenSearch documents index")
    except Exception as error:
        print(f"OpenSearch error: {type(error).__name__}: {error}")
        failures.append("OpenSearch documents index")

    print("\nOVERALL RESULT")
    if failures:
        print(f"FAIL: {', '.join(failures)}")
        raise SystemExit(1)

    print("PASS: RDS tables and OpenSearch are reachable and populated.")


if __name__ == "__main__":
    main()
