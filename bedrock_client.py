
"""Shared Claude-on-Bedrock client with tokenomics logging."""

import contextvars
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import boto3
from dotenv import load_dotenv


load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
TOKENOMICS_LOG_PATH = Path(
    os.getenv("TOKENOMICS_LOG_PATH", "tokenomics_calls.jsonl")
)
INPUT_RATE_PER_MILLION = 3.00
OUTPUT_RATE_PER_MILLION = 15.00

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
_trace_id = contextvars.ContextVar("tokenomics_trace_id", default=None)


def set_trace_id(trace_id: str):
    return _trace_id.set(trace_id)


def reset_trace_id(token) -> None:
    _trace_id.reset(token)


def get_model_id() -> str:
    configured_model = os.getenv("BEDROCK_MODEL_ID")
    if configured_model:
        return configured_model

    account_id = boto3.client("sts", region_name=AWS_REGION).get_caller_identity()[
        "Account"
    ]
    return (
        f"arn:aws:bedrock:{AWS_REGION}:{account_id}:"
        "inference-profile/us.anthropic.claude-sonnet-4-6"
    )


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = input_tokens / 1_000_000 * INPUT_RATE_PER_MILLION
    output_cost = output_tokens / 1_000_000 * OUTPUT_RATE_PER_MILLION
    return round(input_cost + output_cost, 10)


def _extract_text(result: dict[str, Any]) -> str:
    content = result.get("content", [])

    if isinstance(content, str):
        return content.strip()

    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and block.get("text")
    ).strip()


def _append_record(record: dict[str, Any]) -> None:
    TOKENOMICS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TOKENOMICS_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record) + "\n")


def _log_call(
    *,
    component: str,
    model_id: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    status: str = "success",
    error: Optional[str] = None,
) -> float:
    cost_usd = calculate_cost(input_tokens, output_tokens)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "call_id": str(uuid.uuid4()),
        "trace_id": _trace_id.get(),
        "component": component,
        "model_id": model_id,
        "region": AWS_REGION,
        "status": status,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_rate_per_million": INPUT_RATE_PER_MILLION,
        "output_rate_per_million": OUTPUT_RATE_PER_MILLION,
        "cost_usd": cost_usd,
    }

    if error:
        record["error"] = error

    _append_record(record)
    return cost_usd


def _invoke(body: dict[str, Any], component: str) -> dict[str, Any]:
    model_id = get_model_id()

    try:
        response = bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        result = json.loads(response["body"].read())
        usage = result.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        cost_usd = _log_call(
            component=component,
            model_id=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return {
            "result": result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "model_id": model_id,
        }
    except Exception as error:
        _log_call(
            component=component,
            model_id=model_id,
            status="error",
            error=str(error),
        )
        raise


def invoke_claude(
    prompt: str,
    max_tokens: int = 500,
    component: str = "unspecified",
    system: Optional[str] = None,
) -> dict[str, Any]:
    """Call Claude and return text, usage, model, and calculated cost."""
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    if system:
        body["system"] = system

    call = _invoke(body, component)
    result = call["result"]

    return {
        "text": _extract_text(result),
        "input_tokens": call["input_tokens"],
        "output_tokens": call["output_tokens"],
        "cost_usd": call["cost_usd"],
        "model_id": call["model_id"],
        "component": component,
    }


def invoke_claude_raw(
    body: dict[str, Any],
    component: str = "raw",
) -> dict[str, Any]:
    """Call Claude using the full Messages API request shape."""
    request_body = {**body}
    request_body.setdefault("anthropic_version", "bedrock-2023-05-31")
    return _invoke(request_body, component)["result"]
