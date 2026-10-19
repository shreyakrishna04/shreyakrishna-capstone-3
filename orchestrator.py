import re
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.retrieval_agent import run_retrieval_agent
from agents.sql_agent import run_sql_agent
from bedrock_client import invoke_claude


load_dotenv()


class OrchestratorState(TypedDict):
    question: str
    route: Optional[str]
    sql_result: Optional[dict]
    retrieval_result: Optional[dict]
    final_answer: Optional[str]
    token_usage: Optional[dict]


SQL_PATTERNS = [
    r"\bhow many\b",
    r"\bcount\b",
    r"\baverage\b",
    r"\bsum\b",
    r"\btotal\b",
    r"\bhow much\b",
    r"\brevenue\b",
    r"\bbudget\b",
    r"\b(active|completed|open|closed)\s+projects?\b",
    r"\b(list|show)\s+(all\s+)?(clients?|projects?|invoices?|consultants?)\b",
    r"\b(which|what)\s+(clients?|projects?|invoices?|consultants?)\b",
    r"\binvoices?\b.*\b(status|amount|total)\b",
    r"\b(trend|compare)\b.*\b(revenue|budget|invoice|project|client|consultant)\b",
    r"\b(drop|delete|update|insert|alter|truncate|create)\b",
    r"\b(modify|remove)\b.*\b(data|record|row|table)\b",
]


DOC_PATTERNS = [
    r"\bpolicy\b",
    r"\bprocess\b",
    r"\bprocedure\b",
    r"\bmethodology\b",
    r"\bguidelines?\b",
    r"\bapproach\b",
    r"\bplaybook\b",
    r"\bmanual\b",
    r"\bonboarding\b",
    r"\bremote[\s-]work\b",
    r"\bcore[\s-]hours?\b",
    r"\bsummarize\b",
    r"\bexplain\b",
    r"\bdescribe\b",
]

UNKNOWN_TERMS = (
    "weather",
    "forecast",
    "temperature",
    "stock price",
    "breaking news",
    "live traffic",
    "sports score",
    "current news",
)


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def route_question(state: OrchestratorState) -> OrchestratorState:
    question = state["question"].lower().strip()

    if any(term in question for term in UNKNOWN_TERMS):
        route = "unknown"
    else:
        is_sql = matches_any(question, SQL_PATTERNS)
        is_doc = matches_any(question, DOC_PATTERNS)

        if is_sql and is_doc:
            route = "both"
        elif is_sql:
            route = "sql"
        else:
            route = "retrieval"

    return {**state, "route": route}


def call_sql_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in {"sql", "both"}:
        return state

    result = run_sql_agent(state["question"])
    return {**state, "sql_result": result}


def call_retrieval_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in {"retrieval", "both"}:
        return state

    result = run_retrieval_agent(state["question"])
    return {**state, "retrieval_result": result}

UNKNOWN_QUESTION_PATTERN = re.compile(
    r"\b("
    r"weather|forecast|temperature|stock price|breaking news|"
    r"live traffic|sports score|today's news|current news"
    r")\b",
    re.IGNORECASE,
)


def is_unknown_question(question: str) -> bool:
    return bool(UNKNOWN_QUESTION_PATTERN.search(question or ""))


def format_agent_result(agent_name: str, result: dict) -> str:
    return (
        f"{agent_name}:\n"
        f"Success: {result.get('success')}\n"
        f"Answer: {result.get('answer', '')}\n"
        f"Sources: {result.get('sources', [])}\n"
        f"Metadata: {result.get('metadata', {})}"
    )


def synthesise(state: OrchestratorState) -> OrchestratorState:
    context_parts = []
    if state.get("route") == "unknown":
        return {
            **state,
            "final_answer": (
                "I can’t answer that question from the available RDS "
                "and project-document sources."
            ),
        }

    sql_result = state.get("sql_result")
    if sql_result is not None:
        context_parts.append(format_agent_result("SQL Agent Result", sql_result))


    retrieval_result = state.get("retrieval_result")
    if retrieval_result is not None:
        context_parts.append(
            format_agent_result("Retrieval Agent Result", retrieval_result)
        )

    if not context_parts:
        return {
            **state,
            "final_answer": (
                "I was unable to find relevant information "
                "to answer this question."
            ),
            "token_usage": {"input": 0, "output": 0},
        }

    context = "\n\n".join(context_parts)

    prompt = (
        "Answer the question using only the agent results below. "
        "Do not invent information. "
        "If an agent result has success=False, state that its source "
        "could not answer the question. "
        "Cite the relevant agent and source for each factual statement. "
        "Use information from every successful result that is relevant. "
        "Do not mention an agent that was not used.\n\n"
        f"Question: {state['question']}\n\n"
        f"Agent results:\n{context}\n\n"
        "Answer:"
    )
    synthesis_guardrails = """
    Synthesis rules:
    - Use only facts explicitly present in the SQL result or document result.
    - Never invent onboarding steps, policies, counts, or actions.
    - Never claim that data was deleted, updated, inserted, or modified.
    - If a source result begins with INSUFFICIENT_EVIDENCE, state that the
    information is unavailable from the provided sources.
    - Cite SQL results as SQL and document results using their exact filenames.
    - Do not attribute document facts to SQL.
    """
    prompt = f"{prompt}\n\n{synthesis_guardrails}"

    result = invoke_claude(
        prompt,
        max_tokens=800,
        component="synthesis",
    )


    return {
        **state,
        "final_answer": result["text"],
        "token_usage": {
            "input": result["input_tokens"],
            "output": result["output_tokens"],
        },
    }


def build_orchestrator():
    graph = StateGraph(OrchestratorState)

    graph.add_node("router", route_question)
    graph.add_node("sql_agent", call_sql_agent)
    graph.add_node("retrieval_agent", call_retrieval_agent)
    graph.add_node("synthesiser", synthesise)

    graph.set_entry_point("router")
    graph.add_edge("router", "sql_agent")
    graph.add_edge("sql_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", "synthesiser")
    graph.add_edge("synthesiser", END)

    return graph.compile()


orchestrator = build_orchestrator()
