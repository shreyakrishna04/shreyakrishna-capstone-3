import os
import re

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

from bedrock_client import invoke_claude
from onnx_embed import get_embedding


REGION = os.getenv("AWS_REGION", "us-east-1")
INDEX_NAME = os.getenv("OPENSEARCH_INDEX", "documents")
DEFAULT_EXCLUDED_SOURCE = "DPN_Krishna_Resume.pdf"


def get_opensearch_client():
    host = os.environ["OPENSEARCH_ENDPOINT"]
    credentials = boto3.Session().get_credentials()

    if credentials is None:
        raise RuntimeError("AWS credentials were not found.")

    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        REGION,
        "es",
        session_token=credentials.token,
    )

    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )


def get_excluded_sources() -> set[str]:
    configured = os.getenv(
        "RETRIEVAL_EXCLUDED_SOURCES",
        DEFAULT_EXCLUDED_SOURCE,
    )

    return {
        source.strip()
        for source in configured.split(",")
        if source.strip()
    }


def get_min_score() -> float:
    configured = os.getenv("RETRIEVAL_MIN_SCORE", "0.35")

    try:
        return max(0.0, float(configured))
    except ValueError:
        return 0.35


def retrieve(question: str, top_k: int = 5) -> list[dict]:
    query_vector = get_embedding(question)
    client = get_opensearch_client()

    candidate_k = max(top_k * 4, 20)

    response = client.search(
        index=INDEX_NAME,
        body={
            "size": candidate_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": candidate_k,
                    }
                }
            },
        },
    )

    excluded_sources = get_excluded_sources()
    min_score = get_min_score()
    results = []
    seen_chunks = set()
    source_counts = {}

    for hit in response.get("hits", {}).get("hits", []):
        document = hit.get("_source", {})
        source = str(document.get("source", "")).strip()
        content = str(document.get("content", "")).strip()
        score = float(hit.get("_score", 0.0) or 0.0)

        if not source or not content:
            continue

        if source in excluded_sources:
            continue

        if score < min_score:
            continue

        chunk_key = (source, content)

        if chunk_key in seen_chunks:
            continue

        if source_counts.get(source, 0) >= 2:
            continue

        seen_chunks.add(chunk_key)
        source_counts[source] = source_counts.get(source, 0) + 1

        results.append(
            {
                "content": content,
                "source": source,
                "score": score,
            }
        )

        if len(results) >= top_k:
            break

    return results


def generate_cited_answer(
    question: str,
    chunks: list[dict],
) -> tuple[str, dict]:
    context = "\n\n".join(
        f"[Source {index}: {chunk['source']}]\n{chunk['content']}"
        for index, chunk in enumerate(chunks, start=1)
    )

    prompt = (
        "Answer the question using ONLY the context provided below. "
        "Do not use outside knowledge. "
        "If the answer is not in the context, say exactly: "
        "'I cannot find this information in the provided documents.' "
        "For every factual answer, cite the supporting source using the "
        "format [Source N]. Do not cite sources that do not support the answer.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    result = invoke_claude(
        prompt,
        max_tokens=800,  # preserve your existing value
        component="retrieval_generation",
    )

    return result["text"], {
        "input": result["input_tokens"],
        "output": result["output_tokens"],
    }


def check_grounding(answer: str, chunks: list[dict]) -> bool:
    if "cannot find this information" in answer.lower():
        return True

    cited_sources = {
        int(number)
        for number in re.findall(r"\bSources?\s+(\d+)\b", answer, re.IGNORECASE)
    }

    if not cited_sources:
        return False

    return all(1 <= source_number <= len(chunks) for source_number in cited_sources)


def run_retrieval_agent(question: str) -> dict:
    chunks = retrieve(question)

    if not chunks:
        return {
            "success": False,
            "answer": "No relevant documents found.",
            "sources": [],
            "metadata": {
                "chunk_count": 0,
                "grounded": False,
                "token_usage": {"input": 0, "output": 0},
            },
        }

    answer, usage = generate_cited_answer(question, chunks)

    unique_sources = list(dict.fromkeys(chunk["source"] for chunk in chunks))

    return {
        "success": True,
        "answer": answer,
        "sources": unique_sources,
        "metadata": {
            "chunk_count": len(chunks),
            "grounded": check_grounding(answer, chunks),
            "min_score": get_min_score(),
            "excluded_sources": sorted(get_excluded_sources()),
            "token_usage": usage,
        },
    }
