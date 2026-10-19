
from orchestrator import orchestrator
from tokenomics_logger import print_tokenomics_summary, run_traced_query


QUESTIONS = [
    "How many active projects do we have?",
    "What is the total project budget?",
    "List all clients.",
    "What is the average consultant billable hours?",
    "Show invoice statuses and amounts.",
    "What are the remote-work core hours?",
    "Summarize the onboarding process.",
    "Explain the cloud migration playbook.",
    "What methodology is used for project delivery?",
    "How many active projects do we have, and what are the remote-work core hours?",
]


def main() -> None:
    for number, question in enumerate(QUESTIONS, start=1):
        print(f"\n===== QUERY {number} =====")
        print(question)

        result = run_traced_query(orchestrator, question)

        print(f"Route: {result.get('route')}")
        print(f"Answer: {result.get('final_answer', '')}")

    print_tokenomics_summary()


if __name__ == "__main__":
    main()
