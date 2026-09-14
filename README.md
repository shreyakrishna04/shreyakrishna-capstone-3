# Deployment Patterns on AWS

## Table of Contents

- [Deployment Patterns on AWS](#deployment-patterns-on-aws)
  - [Prerequisites](#prerequisites)
  - [Introduction](#introduction)
  - [Setup](#setup)
  - [Table of Contents](#table-of-contents)
  - [Learning Objectives](#learning-objectives)
  - [Why Lambda for This System](#why-lambda-for-this-system)
  - [Part 1: The Lambda Handler](#part-1-the-lambda-handler)
  - [Part 2: Packaging with a Lambda Layer](#part-2-packaging-with-a-lambda-layer)
  - [Part 3: IAM Roles for Production](#part-3-iam-roles-for-production)
  - [Part 4: Deploying the Function](#part-4-deploying-the-function)
  - [Part 5: Testing the Deployed System](#part-5-testing-the-deployed-system)
  - [Resources](#resources)

## Prerequisites

- Building the Orchestrator (this morning)

## Introduction

Everything you've built this week has run in a notebook or a terminal — a real system needs to run without you there to start it. This lesson is about the mechanics of that step: packaging your orchestrator's dependencies, giving it the narrow set of permissions it actually needs (not the broad ones that are easiest to grant), and deploying it somewhere that responds to a request and then goes quiet until the next one. Lambda's request-response shape fits your orchestrator's actual behavior well. This version of the lesson is built specifically around this sandbox's confirmed permissions — a zip-based Lambda deployment with a dependencies layer, which needs no container registry access at all, so every part below is meant to be run hands-on, not just read.

## Setup

**Sandbox notes — read before starting:**

- **RDS works fine here.** Provisioning a `db.t3.micro` instance, connecting to it, and deleting it all work as expected — RDS itself is not restricted in this sandbox. What *is* restricted is attaching the broad `AmazonRDSFullAccess` managed policy to an IAM role (see Part 3) — a narrower, different restriction than "RDS is inaccessible." If you hit an RDS error elsewhere in this course, check first whether you used `admin` as the master username — that's rejected by the Postgres engine itself as a reserved word, independent of any sandbox restriction, and is the most likely cause.
- **IAM role creation works, with real limits** — see Part 3 for exactly what does and doesn't work, and read the warning about these resources being permanent before running anything there.
- **Lambda and CloudWatch Logs permissions haven't been explicitly confirmed for this exact sandbox** the way RDS and IAM have. Part 1's local handler test needs neither, so start there regardless. Treat Part 4's actual `aws lambda create-function` call as the first real test of whether this sandbox's identity can deploy a function at all — if it fails with an authorization error, that's a new, real constraint worth reporting the same way the ECR gap was found, not a sign you did something wrong.

```bash
cd ~/bootcamp/building-the-orchestrator   # wherever this morning's project lives
```

## Learning Objectives

By the end of this lesson, you will be able to:
- Explain why AWS Lambda is an appropriate deployment option for the orchestrator
- Implement a Lambda handler that invokes the LangGraph orchestrator and returns a structured response
- Describe how a Python application's dependencies can be packaged as a Lambda layer, separate from the function's own code
- Apply least-privilege IAM principles when defining permissions for a Lambda-based orchestrator
- Evaluate a Lambda deployment by reviewing its configuration, response, and observability outputs

## Why Lambda for This System

Your orchestrator doesn't need to run continuously — it responds to incoming questions and returns an answer. This request-response pattern with no need for constant uptime is exactly what Lambda is designed for: you pay only for the compute time actually used, and AWS handles all the underlying server management.

The tradeoff: your orchestrator has several dependencies (LangGraph, database drivers, an OpenSearch client, Sentence Transformers) that together exceed the size limit for a Lambda function package deployed directly. Rather than reaching for a container image to solve this, split the dependencies into a **Lambda layer** — a separate zip uploaded once and attached to the function — so the function's own package stays small and fast to update, while the layer holds everything the function imports.

## Part 1: The Lambda Handler

Wrap this morning's orchestrator in a handler function matching the format Lambda expects:

```python
# lambda_handler.py
import json
from orchestrator import build_orchestrator
from tokenomics_logger import run_traced_query  # from this afternoon's earlier lesson

orchestrator = build_orchestrator()

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

        result = run_traced_query(orchestrator, question)

        sql_validation = None
        if result.get("sql_result"):
            sql_validation = result["sql_result"].get("metadata", {}).get("validation")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "answer": result["final_answer"],
                "route": result["route"],
                "sql_validation": sql_validation,
                "token_usage": result.get("token_usage", {})
            })
        }
    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
```

Notice the `sql_validation` field is explicitly surfaced in the response — this is a direct continuation of Day 2's Trust but Verify principle: the validation status should be visible in the system's output, not hidden inside internal logic.

> **Bug this fixes:** an earlier version of this handler called `orchestrator.invoke(...)` directly and imported `log_decision` without ever calling it — meaning the CloudWatch decision-trace claimed in Part 5's check below would never actually appear, since nothing in the handler wrote to it. Calling `run_traced_query` (from this afternoon's earlier tokenomics lesson) instead of the bare `orchestrator.invoke` is what actually produces that trace.

> **✅ Check:** call the handler directly with a mock event, no Lambda required:
> ```python
> from lambda_handler import handler
> event = {"body": '{"question": "How many active projects do we have?"}'}
> print(handler(event, None))
> ```
> Confirm you get back a 200 status with `answer`, `route`, `sql_validation`, and `token_usage` all populated — this is the actual logic Lambda would run, so testing it this way catches real bugs even before anything touches AWS.

## Part 2: Packaging with a Lambda Layer

Keep the function package itself small — just your own code — and put every dependency into one or more separate layers.

`sentence-transformers` (used by your Retrieval Agent for embeddings) pulls in PyTorch, which is large enough to risk exceeding a Lambda layer's 250MB unzipped limit on its own. Split dependencies across two layers rather than one, so a size problem in the heavy layer doesn't block everything else:

```bash
# Layer 1: everything except sentence-transformers
mkdir -p layer-core/python
pip install langgraph boto3 psycopg2-binary opensearch-py requests-aws4auth python-dotenv \
  -t layer-core/python --break-system-packages
cd layer-core && zip -r ../orchestrator-core-layer.zip . && cd ..

aws lambda publish-layer-version \
  --layer-name orchestrator-core-deps \
  --zip-file fileb://orchestrator-core-layer.zip \
  --compatible-runtimes python3.11

# Layer 2: sentence-transformers, kept separate because of its size
mkdir -p layer-embeddings/python
pip install sentence-transformers -t layer-embeddings/python --break-system-packages
cd layer-embeddings && zip -r ../orchestrator-embeddings-layer.zip . && cd ..

aws lambda publish-layer-version \
  --layer-name orchestrator-embeddings-deps \
  --zip-file fileb://orchestrator-embeddings-layer.zip \
  --compatible-runtimes python3.11
```

Note the returned `LayerVersionArn` from **both** commands — you'll attach both to the function in Part 4.

```bash
# Package just your own code — no dependencies bundled here
mkdir -p function_package
cp lambda_handler.py orchestrator.py tokenomics_logger.py function_package/
cp -r agents/ function_package/

cd function_package && zip -r ../orchestrator_function.zip . && cd ..
```

> **⚠️ Common issue:** `psycopg2` build failures are almost always solved by using `psycopg2-binary` in the pip install above, not `psycopg2` — the binary package avoids native dependency compilation, which matters here since the layer is built on your local machine, not necessarily the same OS/architecture Lambda runs on. If you hit an import error at runtime specifically (not at build time), rebuild the layer using the `--platform manylinux2014_x86_64 --only-binary=:all:` pip flags to force Lambda-compatible wheels.

> **⚠️ Common issue:** if the embeddings layer still exceeds the 250MB unzipped limit after being split out on its own, that's a real constraint worth reporting rather than working around blindly — a common production answer is moving embedding generation to its own dedicated service rather than bundling it into the same function as the orchestrator, but that's a bigger architectural change than this lesson expects you to make today.

## Part 3: IAM Roles for Production

**Run this part for real — with two firm limits, confirmed in this sandbox:**

1. **Attaching the three `*FullAccess` managed policies below will fail** with `AccessDenied` — this identity can only attach a narrow, pre-approved set of managed policies (the basic Lambda execution role used in the first `attach-role-policy` call works fine). Use the scoped inline policy shown further down instead — it actually works, and it's a better answer anyway (see the note below on why).
2. **Whatever role and policy you create here, you cannot delete.** This identity has `iam:CreateRole` and `iam:PutRolePolicy` but not `iam:DeleteRolePolicy`, and a role can't be deleted while it still has a policy attached — so once you create it, it's permanent for the life of this AWS account. **Use a role name that includes your own name or initials** (e.g. `orchestrator-lambda-role-<yourname>`) so instructors can tell whose is whose later, and only create it once — don't experiment by creating and re-creating.

Following the least-privilege principle from Module 3 Day 1's IAM lesson, scope this role to exactly what the orchestrator needs — no more.

```bash
cat > trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

aws iam create-role --role-name orchestrator-lambda-role-<yourname> --assume-role-policy-document file://trust.json

aws iam attach-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

The three commands below are what a real deployment would need for Bedrock, OpenSearch, and RDS access — **explain, don't run these specific ones**, since the managed `*FullAccess` policies are denied in this sandbox:

```bash
aws iam attach-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
aws iam attach-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-arn arn:aws:iam::aws:policy/AmazonOpenSearchServiceFullAccess
aws iam attach-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess
```

> **What actually works here, and why it's arguably the better lesson anyway:** a scoped **inline** policy via `put-role-policy` succeeds in this sandbox where the broad managed policies don't. This is a real, runnable stand-in — and it's exactly the least-privilege pattern the note below already says a production engagement should use instead of `*FullAccess`:
> ```bash
> cat > scoped-policy.json <<'EOF'
> {"Version":"2012-10-17","Statement":[
>   {"Effect":"Allow","Action":"bedrock:InvokeModel","Resource":"*"}
> ]}
> EOF
> aws iam put-role-policy --role-name orchestrator-lambda-role-<yourname> --policy-name orchestrator-inline-policy --policy-document file://scoped-policy.json
> ```
> Remember: this inline policy is also permanent once created (no `iam:DeleteRolePolicy` here) — write it once, get it right, don't iterate by trial and error.

> **Note:** The `*FullAccess` managed policies above are what many teams reach for "for bootcamp simplicity," and it's worth understanding why they're tempting: one line, no need to enumerate exact actions. But a real production engagement should write custom policies scoped to the specific resources (a single OpenSearch domain, a single RDS instance) rather than full-access managed policies — the same least-privilege principle from Day 1. This sandbox's restriction, ironically, forces the stricter and more correct answer.

## Part 4: Deploying the Function

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/orchestrator-lambda-role-<yourname>"
CORE_LAYER_ARN="<paste the LayerVersionArn from the core-deps layer here>"
EMBEDDINGS_LAYER_ARN="<paste the LayerVersionArn from the embeddings layer here>"

aws lambda create-function \
  --function-name multi-agent-orchestrator \
  --runtime python3.11 \
  --handler lambda_handler.handler \
  --zip-file fileb://orchestrator_function.zip \
  --role "$ROLE_ARN" \
  --layers "$CORE_LAYER_ARN" "$EMBEDDINGS_LAYER_ARN" \
  --timeout 120 \
  --memory-size 512 \
  --environment "Variables={AWS_REGION=us-east-1,OPENSEARCH_ENDPOINT=your_endpoint,OPENSEARCH_INDEX=documents,RDS_HOST=your_host,RDS_PORT=5432,RDS_DATABASE=bootcamp,RDS_USER=dbadmin,RDS_PASSWORD=your_password}"
```

> **Note what's *not* in that environment variable list:** no `ANTHROPIC_API_KEY`. Nothing in this orchestrator calls anything but `bedrock_client.py`'s `invoke_claude()`, which authenticates through the Lambda's own IAM role, not an API key — there's no code path anywhere in this stack that would ever read that variable.

> **⚠️ Don't use `admin` as `RDS_USER`.** RDS rejects it outright as the master username for the `postgres` engine — `MasterUsername admin cannot be used as it is a reserved word used by the engine`. Use `dbadmin` (shown above) or any other non-reserved name, matching whatever you set during RDS provisioning.

> **⚠️ Common issue:** Lambda timeout on multi-agent queries, especially "both"-routed questions calling two agents plus synthesis. The default 3-second timeout will fail almost every time — 120 seconds gives enough headroom.

> **⚠️ If this specific command fails with an authorization error:** that's new, useful information about this sandbox's actual permissions, not a mistake in the steps above — note exactly which action was denied (it'll be in the error message) and flag it the same way the ECR restriction was found, so this lesson can be corrected again if needed.

> **If you update your code after this initial deploy**, use `aws lambda update-function-code --function-name multi-agent-orchestrator --zip-file fileb://orchestrator_function.zip` — you only need to touch the layers again if your dependencies themselves change, not your own code.

## Part 5: Testing the Deployed System

```bash
cat > test_event.json <<'EOF'
{"body": "{\"question\": \"How many active projects do we have?\"}"}
EOF

aws lambda invoke \
  --function-name multi-agent-orchestrator \
  --payload file://test_event.json \
  --cli-binary-format raw-in-base64-out \
  response.json

cat response.json
aws logs tail /aws/lambda/multi-agent-orchestrator --follow
```

> **✅ Check:** Confirm your response includes `answer`, `route`, `sql_validation`, and `token_usage` fields, and that CloudWatch Logs show the decision trace from Part 1's `run_traced_query` call. Compare this against what you already saw locally in Part 1 — same response shape, same trace, now coming from an actual deployed Lambda instead of a direct function call. This is the same deployment your capstone builds on directly.

## Resources

- [AWS Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/chapter-layers.html)
- [AWS Lambda Deployment Packages (.zip)](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
