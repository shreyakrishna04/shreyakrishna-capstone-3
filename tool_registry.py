
from agents.retrieval_agent import run_retrieval_agent
from agents.sql_agent import run_sql_agent


TOOL_REGISTRY = {
    "sql_agent": {
        "function": run_sql_agent,
        "description": (
            "Use for structured questions answerable from the RDS tables "
            "clients, projects, consultants, and invoices. Do not use for "
            "policies, procedures, resumes, playbooks, or document content."
        ),
    },
    "retrieval_agent": {
        "function": run_retrieval_agent,
        "description": (
            "Use for policies, procedures, guidelines, resumes, playbooks, "
            "and other unstructured document questions. Do not use for "
            "counts, aggregations, trends, or queries requiring RDS tables."
        ),
    },
}
