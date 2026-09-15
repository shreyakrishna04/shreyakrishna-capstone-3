"""
orchestrator.py -- routes a question to the SQL agent, the retrieval agent,
or both, then synthesizes a single cited answer.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from bedrock_client import invoke_claude
from agents.sql_agent import run_sql_agent
from agents.retrieval_agent import run_retrieval_agent
from dotenv import load_dotenv
load_dotenv()


class OrchestratorState(TypedDict):
    question: str
    route: Optional[str]
    sql_result: Optional[dict]
    retrieval_result: Optional[dict]
    final_answer: Optional[str]
    token_usage: Optional[dict]


SQL_KEYWORDS = [
    "how many", "count", "total", "average", "sum", "revenue",
    "how much", "list all", "which clients", "projects", "invoices",
    "consultants", "compare", "trend", "budget"
]
DOC_KEYWORDS = [
    "policy", "process", "procedure", "methodology", "what is",
    "explain", "describe", "guidelines", "approach", "onboarding"
]


def route_question(state: OrchestratorState) -> OrchestratorState:
    question = state["question"].lower()
    is_sql = any(k in question for k in SQL_KEYWORDS)
    is_doc = any(k in question for k in DOC_KEYWORDS)

    if is_sql and is_doc:
        route = "both"
    elif is_sql:
        route = "sql"
    else:
        route = "retrieval"

    return {**state, "route": route}


def call_sql_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in ["sql", "both"]:
        return state
    result = run_sql_agent(state["question"])
    return {**state, "sql_result": result}


def call_retrieval_agent(state: OrchestratorState) -> OrchestratorState:
    if state["route"] not in ["retrieval", "both"]:
        return state
    result = run_retrieval_agent(state["question"])
    return {**state, "retrieval_result": result}


def synthesise(state: OrchestratorState) -> OrchestratorState:
    context = ""

    if state.get("sql_result") and state["sql_result"].get("success"):
        r = state["sql_result"]
        context += f"SQL Agent Result:\n{r['answer']}\nSources: {r['sources']}\n\n"

    if state.get("retrieval_result") and state["retrieval_result"].get("success"):
        r = state["retrieval_result"]
        context += f"Retrieval Agent Result:\n{r['answer']}\nSources: {r['sources']}\n\n"

    if not context:
        return {
            **state,
            "final_answer": "I was unable to find relevant information to answer this question.",
            "token_usage": {"input": 0, "output": 0}
        }

    prompt = (
        f"Using the information below, answer the following question clearly.\n"
        f"Cite which agent(s) and source(s) the information came from.\n\n"
        f"Question: {state['question']}\n\n"
        f"Information:\n{context}"
    )
    result = invoke_claude(prompt, max_tokens=800)

    return {
        **state,
        "final_answer": result["text"],
        "token_usage": {
            "input": result["input_tokens"],
            "output": result["output_tokens"]
        }
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
