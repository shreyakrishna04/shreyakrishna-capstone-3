# Deployment Patterns on AWS

**Session Time:** 60 minutes

## Prerequisites

- Building the Orchestrator (this morning)
- Docker installed and running

## Setup

**Sandbox note — read this before Part 2:** this environment's IAM identity does not have `ecr:CreateRepository` — there is no way to push a Docker image anywhere Lambda could pull it from. That means **Parts 2, 4, and 5 cannot be run hands-on by students in this sandbox** — the same category of environment gap as the RDS block on Day 1 (still pending as of this writing), just on the deployment side instead of the database side. IAM role creation, however, *does* work here (with limits) — see Part 3.

This isn't a reason to skip the lesson — it's a reason to teach it differently:

- **Part 1 (the Lambda handler) is fully hands-on.** It's plain Python — write `lambda_handler.py`, and test it by calling `handler(event, context)` directly with a mock `event` dict, no real Lambda, ECR, or IAM involved. Do this part for real.
- **Part 3 (IAM roles) is hands-on, with real limits.** `iam:CreateRole` and a scoped inline policy both work — see Part 3's callout for exactly what runs and what doesn't, and read the warning about these resources being permanent before running anything.
- **Parts 2 and 4 (Docker/ECR packaging, deploying the function) should be walked through as instructor-led explanation, not a hands-on exercise.** Read every command below and understand what it does and why — the concepts (why a container image instead of a zip package, what a Lambda deployment actually needs) are the point of this lesson regardless of whether the commands can run here.
- **Part 5 (testing the deployed system)** depends on Part 4 having actually deployed something — treat it as "here's how you'd verify this if you had a real deployment," not a live exercise.

If GA's platform later grants `ecr:CreateRepository`/`ecr:PutImage`, revisit Parts 2, 4, and 5 as fully hands-on — nothing about the code below would need to change, only the sandbox's permissions.

## Table of Contents

- [Deployment Patterns on AWS](#deployment-patterns-on-aws)
  - [Prerequisites](#prerequisites)
  - [Table of Contents](#table-of-contents)
  - [Learning Objectives](#learning-objectives)
  - [Why Lambda for This System](#why-lambda-for-this-system)
  - [Part 1: The Lambda Handler](#part-1-the-lambda-handler)
  - [Part 2: Packaging with Docker](#part-2-packaging-with-docker)
  - [Part 3: IAM Roles for Production](#part-3-iam-roles-for-production)
  - [Part 4: Deploying the Function](#part-4-deploying-the-function)
  - [Part 5: Testing the Deployed System](#part-5-testing-the-deployed-system)
  - [Resources](#resources)

## Learning Objectives

- Explain why Lambda is a good fit for this orchestrator's deployment
- Wrap a LangGraph orchestrator in a Lambda handler function
- Package a Python application with dependencies as a Docker container image for Lambda
- Configure least-privilege IAM roles for a production Lambda deployment
- Deploy and test the function with a live invocation

## Why Lambda for This System

Your orchestrator doesn't need to run continuously — it responds to incoming questions and returns an answer. This request-response pattern with no need for constant uptime is exactly what Lambda is designed for: you pay only for the compute time actually used, and AWS handles all the underlying server management.

The tradeoff, and why we're using a **container image** rather than a plain zip-file Lambda deployment: your orchestrator has several dependencies (LangGraph, Anthropic SDK, database drivers, OpenSearch client) that together exceed the size limits of a standard zip-based Lambda package. Container images support a much larger deployment size, which is why this is the standard pattern for dependency-heavy Python Lambda functions.

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

> **✅ Check (do this now, hands-on):** call the handler directly with a mock event, no Lambda required:
> ```python
> from lambda_handler import handler
> event = {"body": '{"question": "How many active projects do we have?"}'}
> print(handler(event, None))
> ```
> Confirm you get back a 200 status with `answer`, `route`, `sql_validation`, and `token_usage` all populated — this is the actual logic Lambda would run, so testing it this way catches real bugs even though it never touches AWS.

## Part 2: Packaging with Docker

> **⚠️ Explain, don't run:** `docker build` works fine locally (Docker itself isn't restricted), but `aws ecr create-repository` and the subsequent `docker push` will fail — this sandbox's IAM identity doesn't have `ecr:CreateRepository`. There's nowhere to push the image to. Read through this section to understand what packaging a Python app for Lambda actually involves; don't expect the `aws ecr`/`docker push` commands below to succeed.

```dockerfile
FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt .
RUN pip install -r requirements.txt --target .
COPY . .
CMD ["lambda_handler.handler"]
```

```bash
docker build -t multi-agent-orchestrator .

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com"

aws ecr create-repository --repository-name multi-agent-orchestrator --region us-east-1

docker tag multi-agent-orchestrator:latest "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/multi-agent-orchestrator:latest"
docker push "$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/multi-agent-orchestrator:latest"
```

> **⚠️ Common issue:** Build failures related to `psycopg2` are almost always solved by using `psycopg2-binary` in `requirements.txt` instead of `psycopg2` — the binary package avoids native dependency compilation inside the Lambda container image.

## Part 3: IAM Roles for Production

> **✅ Run this part for real — with two firm limits.** Role creation itself works in this sandbox, so do this hands-on rather than just reading it. But:
> 1. **Attaching the three `*FullAccess` managed policies below will fail** with `AccessDenied` — this identity can only attach a narrow, pre-approved set of managed policies (the basic Lambda execution role used in the first `attach-role-policy` call works fine). Use the scoped inline policy shown further down instead — it actually works, and it's a better answer anyway (see the note below on why).
> 2. **Whatever role and policy you create here, you cannot delete.** This identity has `iam:CreateRole` and `iam:PutRolePolicy` but not `iam:DeleteRolePolicy`, and a role can't be deleted while it still has a policy attached — so once you create it, it's permanent for the life of this AWS account. **Use a role name that includes your own name or initials** (e.g. `orchestrator-lambda-role-<yourname>`) so instructors can tell whose is whose later, and only create it once — don't experiment by creating and re-creating.

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

> **⚠️ Explain, don't run:** this depends on Part 2's image already existing in ECR, which isn't possible here. Read through the `aws lambda create-function` call and understand what each flag configures (image location, execution role, timeout, memory, environment variables) — this is exactly what the capstone's Lambda deployment does, so understanding it now matters even without running it today.

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/multi-agent-orchestrator:latest"
ROLE_ARN="arn:aws:iam::$ACCOUNT_ID:role/orchestrator-lambda-role"

aws lambda create-function \
  --function-name multi-agent-orchestrator \
  --package-type Image \
  --code ImageUri="$IMAGE_URI" \
  --role "$ROLE_ARN" \
  --timeout 120 \
  --memory-size 512 \
  --environment Variables="{ANTHROPIC_API_KEY=your_key,AWS_REGION=us-east-1,OPENSEARCH_ENDPOINT=your_endpoint,OPENSEARCH_INDEX=documents,RDS_HOST=your_host,RDS_PORT=5432,RDS_DATABASE=bootcamp,RDS_USER=admin,RDS_PASSWORD=your_password}"
```

> **⚠️ Common issue:** Lambda timeout on multi-agent queries, especially "both"-routed questions calling two agents plus synthesis. The default 3-second timeout will fail almost every time — 120 seconds gives enough headroom.

## Part 5: Testing the Deployed System

> **⚠️ Explain, don't run:** there's no deployed function to test against in this sandbox. This is "here's how you'd verify a real deployment," not a live exercise today — the pattern (invoke, check the response fields, tail the logs) is exactly what you'd do once the capstone or a future engagement has real ECR/Lambda access.

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

> **✅ Check (what you can actually verify today):** you already confirmed the `answer`, `route`, `sql_validation`, and `token_usage` fields and the decision-trace logging back in Part 1's local `handler(event, None)` test — that's the same response shape and the same logging a real deployed Lambda would produce. This is the same deployment your capstone will build on directly, once ECR/Lambda access is available.

## Resources

- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
