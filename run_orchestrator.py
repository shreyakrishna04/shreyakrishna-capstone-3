
import sys
from pprint import pprint
from orchestrator import build_orchestrator


def main():
    question = " ".join(sys.argv[1:]).strip()

    if not question:
        question = "How many active projects do we have?"

    app = build_orchestrator()

    initial_state = {
        "question": question,
        "route": None,
        "sql_result": None,
        "retrieval_result": None,
        "final_answer": None,
        "token_usage": None,
    }

    result = app.invoke(initial_state)

    print("\nRoute:", result.get("route"))
    print("\nFinal answer:")
    print(result.get("final_answer"))

    print("\nComplete state:")
    pprint(result)


if __name__ == "__main__":
    main()
