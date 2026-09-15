# Deployment Patterns on AWS

## Prerequisites

- Building the Orchestrator (this morning)

## Introduction

Everything you've built this week has run in a notebook or a terminal — a real system needs to run without you there to start it. This lesson covers packaging your orchestrator and its dependencies, giving it exactly the permissions it needs, provisioning the real infrastructure it depends on, and deploying it somewhere that responds to a request and goes quiet until the next one.

## What's Actually Confirmed About This Sandbox

| Action | Status |
|---|---|
| `lambda:CreateFunction` (zip, direct upload or from S3) | **Works** |
| `lambda:InvokeFunction`, `lambda:DeleteFunction` | **Works** |
| `lambda:PublishLayerVersion` | **Blocked** |
| `ecr:CreateRepository` / `DescribeRepositories` | **Blocked** |
| `iam:CreateRole`, `AttachRolePolicy` (basic execution policy), `PutRolePolicy` (inline) | **Works** |
| `iam:AttachRolePolicy` (`*FullAccess` managed policies) | **Blocked** |
| `iam:DeleteRolePolicy` / role deletion | **Blocked — permanent once created** |
| `s3:CreateBucket`, upload, read | **Works** |
| RDS and OpenSearch provisioning, connecting, deleting | **Works** |

**What follows from this:** no layers, no container images — one combined deployment package, deployed via S3 (it will be well over the 50MB direct-upload limit once dependencies are included). Use a role name that includes your own name or initials — it's permanent for the life of this account.

## Learning Objectives

By the end of this lesson, you will be able to:
- Package a Python application's dependencies into a single Lambda deployment package, working within real size constraints
- Replace a dependency too large to deploy (PyTorch) with a smaller, equivalent alternative (ONNX Runtime)
- Provision and connect a Lambda function to real RDS and OpenSearch infrastructure
- Apply least-privilege IAM principles when defining permissions for a Lambda-based orchestrator
- Tear down every provisioned resource when finished, to avoid ongoing cost

## Files in This Lesson

```
bedrock_client.py          -- shared Claude-via-Bedrock helper
onnx_embed.py               -- embeddings via ONNX Runtime (see Part 2 for why)
orchestrator.py             -- the LangGraph orchestrator
tokenomics_logger.py        -- logging and tracing
lambda_handler.py           -- the actual Lambda entry point
teardown.sh                 -- deletes every resource this lesson creates
agents/
  __init__.py
  sql_agent.py               -- self-contained validated SQL agent
  retrieval_agent.py         -- self-contained retrieval agent
```

Not included, download separately in Part 2: `model.onnx` and `tokenizer.json`.

## Part 1: The Lambda Handler

`lambda_handler.py` wraps the orchestrator in the format Lambda expects, calling `run_traced_query` so decision logging actually happens. `tokenomics_logger.py` writes its log file to `/tmp/agent_decisions.jsonl`, not a relative path — **Lambda's deployment package directory is read-only at runtime; only `/tmp` is writable.**

> **✅ Check — do this before anything AWS-related:**
> ```python
> from lambda_handler import handler
> event = {"body": '{"question": "How many active projects do we have?"}'}
> print(handler(event, None))
> ```
> Needs your actual RDS, OpenSearch, and Bedrock access configured locally first.

## Part 2: Packaging

### Why not `sentence-transformers`

`agents/retrieval_agent.py` needs to embed text for OpenSearch queries. `sentence-transformers` depends on PyTorch — and PyTorch's core runtime library alone (`libtorch_cpu.so`) is about 418MB, already over Lambda's 250MB unzipped limit by itself, before anything else is added.

`onnx_embed.py` reimplements the same `all-MiniLM-L6-v2` pipeline using `onnxruntime` and `tokenizers` instead — a much smaller footprint that produces numerically equivalent embeddings.

```bash
mkdir -p ~/deployment-package
curl -L -o ~/deployment-package/model.onnx "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
curl -L -o ~/deployment-package/tokenizer.json "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
```

### Building the package

```bash
cd ~/deployment-package

pip install langgraph psycopg2-binary opensearch-py requests-aws4auth python-dotenv \
  onnxruntime tokenizers numpy \
  langchain-core pydantic pydantic-core langsmith \
  langgraph-sdk ormsgpack orjson httpx httpcore truststore tenacity uuid_utils \
  -t . --break-system-packages

cp /path/to/bedrock_client.py /path/to/orchestrator.py /path/to/tokenomics_logger.py \
   /path/to/lambda_handler.py /path/to/onnx_embed.py .
cp -r /path/to/agents .
```

> **Install every package in that second line, even though nothing in this codebase imports them directly.** `langgraph` requires `langchain-core` at graph-compile time, `langchain-core` requires `langsmith` at import time (including its `_openapi_client` submodule), and `langgraph`'s checkpoint serialization requires `ormsgpack`. All are genuine runtime dependencies, not optional extras.

**Trim what's confirmed unused:**

```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -maxdepth 1 -name "*.dist-info" -exec rm -rf {} +
find . -type d -name "tests" -exec rm -rf {} + 2>/dev/null
find . -type d -name "test" -exec rm -rf {} + 2>/dev/null

# huggingface_hub/hf_xet: support downloading FROM the HF Hub at runtime.
# onnx_embed.py loads model.onnx/tokenizer.json from local files -- never
# calls the Hub.
rm -rf huggingface_hub huggingface_hub-*.dist-info hf_xet hf_xet-*.dist-info

# langgraph_sdk's own remote-hosted-service transport -- this codebase only
# runs graphs locally.
rm -rf websockets langchain_protocol google bin

# LangSmith's remote tracing transport specifically -- not langsmith itself
# (required, see above), just its compression/gRPC dependencies.
rm -rf zstandard zstandard-*.dist-info grpc grpc-*.dist-info

# numpy submodules onnx_embed.py's actual usage never touches:
rm -rf numpy/f2py numpy/ma numpy/polynomial numpy/random/_examples numpy/matlib.py numpy/matlib.pyi

# onnxruntime's own bundled sample/test .onnx files:
rm -rf onnxruntime/datasets

du -sb . | awk '{printf "%.2f MB\n", $1/1024/1024}'
```

This should land under 250MB with some headroom.

> **⚠️ Verify with a real invocation, not just an import.** `import langgraph` succeeding doesn't prove `StateGraph(...).compile()` will work, and `build_orchestrator()` succeeding doesn't prove `.invoke()` will. Always confirm with:
> ```bash
> PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -c "
> import sys
> sys.path.insert(0, 'agents')
> from orchestrator import build_orchestrator
> graph = build_orchestrator()
> result = graph.invoke({
>     'question': 'test question', 'route': None, 'sql_result': None,
>     'retrieval_result': None, 'final_answer': None, 'token_usage': None
> })
> " 2>&1 | tail -5
> ```
> Expect this to fail with an OpenSearch connection error (no real endpoint configured yet) — that's the correct result at this stage, confirming everything upstream of real infrastructure works. Any `ModuleNotFoundError` instead means a real dependency gap.

> **⚠️ `-t .` doesn't overwrite existing files.** If you need to fully reinstall a package that's already partially present, `rm -rf <package-dir>` first, then reinstall into the empty space — otherwise pip silently skips writing anything.

```bash
cd ~/deployment-package
zip -r ../orchestrator_function.zip . -x "*.git*"
```

## Part 3: IAM Roles for Production

Only `AWSLambdaBasicExecutionRole` and an inline policy work here — every `*FullAccess` managed policy attachment is denied. Whatever role and policy you create is permanent, so name it with your own initials.

```bash
cat > trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

aws iam create-role --role-name orchestrator-lambda-role-<yourname> --assume-role-policy-document file://trust.json
aws iam attach-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

cat > scoped-policy.json <<'EOF'
{"Version":"2012-10-17","Statement":[
  {"Effect":"Allow","Action":"bedrock:InvokeModel","Resource":"*"}
]}
EOF
aws iam put-role-policy \
  --role-name orchestrator-lambda-role-<yourname> \
  --policy-name orchestrator-inline-policy \
  --policy-document file://scoped-policy.json
```

## Part 4: Provisioning Real Infrastructure

### RDS

```bash
aws rds create-db-instance \
  --db-instance-identifier orchestrator-db-<yourname> \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --engine-version 16 \
  --master-username dbadmin \
  --master-user-password '<a-real-password>' \
  --allocated-storage 20 \
  --db-name bootcamp \
  --publicly-accessible \
  --backup-retention-period 0

aws rds wait db-instance-available --db-instance-identifier orchestrator-db-<yourname>

SG_ID=$(aws rds describe-db-instances --db-instance-identifier orchestrator-db-<yourname> \
  --query "DBInstances[0].VpcSecurityGroups[0].VpcSecurityGroupId" --output text)
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 5432 --cidr 0.0.0.0/0
```

> **⚠️ Use `dbadmin`, never `admin`** — RDS rejects `admin` outright as a reserved word for the `postgres` engine.

Create the schema (`clients`, `projects`, `consultants`, `invoices` — the schema `agents/sql_agent.py`'s `SCHEMA_CONTEXT` assumes) and seed it, then connect with `sslmode=require` — RDS's default parameter group rejects unencrypted connections.

### OpenSearch

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

python3 -c "
import json, sys
policy = {
  'Version': '2012-10-17',
  'Statement': [{
    'Effect': 'Allow',
    'Principal': {'AWS': f'arn:aws:iam::${ACCOUNT_ID}:role/orchestrator-lambda-role-<yourname>'},
    'Action': 'es:*',
    'Resource': f'arn:aws:es:us-east-1:${ACCOUNT_ID}:domain/orchestrator-search-<yourname>/*'
  }]
}
sys.stdout.write(json.dumps(policy))
" > opensearch-access-policy.json
```

> **⚠️ Use `sys.stdout.write()`, not `print()`, when writing this file.** `print()` appends a trailing newline, which fails `--access-policies`' validation with a confusing generic error.

```bash
aws opensearch create-domain \
  --domain-name orchestrator-search-<yourname> \
  --engine-version OpenSearch_2.11 \
  --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
  --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=10 \
  --access-policies file://opensearch-access-policy.json \
  --domain-endpoint-options EnforceHTTPS=true
```

This takes 15-20 minutes. Check with `aws opensearch describe-domain --domain-name orchestrator-search-<yourname> --query "DomainStatus.Processing"` — `False` means ready.

**Create the index with `space_type: "innerproduct"`, not `"cosinesimil"`:**

```python
index_body = {
    "settings": {"index.knn": True},
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "source": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {"name": "hnsw", "space_type": "innerproduct", "engine": "faiss"}
            }
        }
    }
}
```

> **⚠️ FAISS's `hnsw` method does not support `cosinesimil`** — only `nmslib` and `lucene` engines do. Since `onnx_embed.py`'s output is L2-normalized, cosine similarity and inner product are mathematically identical for unit-length vectors, so `innerproduct` gives the same ranking `cosinesimil` would have.

## Part 5: Deploying and Testing

```bash
BUCKET="lambda-deploy-${ACCOUNT_ID}-<yourname>"
aws s3 mb "s3://${BUCKET}"
aws s3 cp ~/orchestrator_function.zip "s3://${BUCKET}/orchestrator_function.zip"

aws lambda create-function \
  --function-name multi-agent-orchestrator \
  --runtime python3.12 \
  --handler lambda_handler.handler \
  --code S3Bucket="${BUCKET}",S3Key=orchestrator_function.zip \
  --role "arn:aws:iam::${ACCOUNT_ID}:role/orchestrator-lambda-role-<yourname>" \
  --timeout 120 \
  --memory-size 512 \
  --environment "Variables={OPENSEARCH_ENDPOINT=<your-domain-endpoint>,RDS_HOST=<your-rds-endpoint>,RDS_PORT=5432,RDS_DATABASE=bootcamp,RDS_USER=dbadmin,RDS_PASSWORD=<your-password>}"
```

> **⚠️ Match `--runtime` to whatever Python version built your package.** Compiled extensions (`.so` files) are not binary-compatible across Python minor versions. Check `python3 --version` and every `.so` filename in your package (`find . -name "*.so"`) before choosing — a mismatch produces a confusing `ModuleNotFoundError` for a package that's genuinely present.

> **⚠️ After any `update-function-code`, wait for real completion before testing:**
> ```bash
> aws lambda wait function-updated-v2 --function-name multi-agent-orchestrator
> ```
> A fixed `sleep` duration isn't reliable — Lambda's propagation across execution environments can outlast it, producing inconsistent results between invocations.

```bash
echo '{"body": "{\"question\": \"How many active projects do we have, and what is our onboarding policy?\"}"}' > test_both.json
aws lambda invoke --function-name multi-agent-orchestrator --payload file://test_both.json \
  --cli-binary-format raw-in-base64-out response.json
cat response.json
```

> **✅ What success looks like:** `"route": "both"`, `"sql_validation": "PASSED"`, an answer that correctly combines a real RDS query result with a real OpenSearch-retrieved document, attributing each part to its source agent, and non-zero `token_usage`.

## Part 6: Tear Down When Finished

RDS and OpenSearch both bill continuously while running — tear down as soon as you're done testing. See `teardown.sh`, included alongside this README.

```bash
chmod +x teardown.sh
./teardown.sh --name <yourname>              # dry run -- lists what would be deleted
./teardown.sh --name <yourname> --confirm    # actually deletes
```

`--name` is required — it's how the script identifies which resources are yours, matching whatever suffix you chose in Parts 3-4. Running it without `--name` prints usage instructions and exits.

`--confirm` isn't the final safety gate — you'll be prompted to type `DELETE` (all capital letters) before anything actually happens.

**One thing it cannot do:** delete the IAM role or its inline policy — `iam:DeleteRolePolicy` is blocked in this sandbox, and a role can't be deleted while a policy is attached. This isn't a cost concern; IAM itself is free.

**One thing it doesn't wait for:** RDS and OpenSearch deletions don't finish the moment the script returns. Confirm they're genuinely gone afterward:

```bash
aws rds describe-db-instances --db-instance-identifier orchestrator-db-<yourname>
aws opensearch describe-domain --domain-name orchestrator-search-<yourname>
```

Both should eventually return a "not found" style error once deletion is fully complete.

## Resources

- [AWS Lambda Deployment Packages (.zip)](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [OpenSearch k-NN methods and engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)
