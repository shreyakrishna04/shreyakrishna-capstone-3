Here's the document with every PowerShell block converted to bash (Ubuntu):

---

# Deployment Patterns on AWS

**Session Time:** 60 minutes

## Prerequisites

- Building the Orchestrator (this morning)
- Docker installed and running

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
from tokenomics_logger import log_decision  # from this afternoon's earlier lesson

orchestrator = build_orchestrator()

def handler(event, context):
    try:
        body = json.loads(event.get("body", "{}"))
        question = body.get("question", "").strip()

        if not question:
            return {"statusCode": 400, "body": json.dumps({"error": "question is required"})}

        result = orchestrator.invoke({
            "question": question, "route": None, "sql_result": None,
            "retrieval_result": None, "final_answer": None, "token_usage": None
        })

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

## Part 2: Packaging with Docker

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

Following the least-privilege principle from Module 3 Day 1's IAM lesson, scope this role to exactly what the orchestrator needs — no more.

```bash
cat > trust.json <<'EOF'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}
EOF

aws iam create-role --role-name orchestrator-lambda-role --assume-role-policy-document file://trust.json

aws iam attach-role-policy --role-name orchestrator-lambda-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam attach-role-policy --role-name orchestrator-lambda-role --policy-arn arn:aws:iam::aws:policy/AmazonBedrockFullAccess
aws iam attach-role-policy --role-name orchestrator-lambda-role --policy-arn arn:aws:iam::aws:policy/AmazonOpenSearchServiceFullAccess
aws iam attach-role-policy --role-name orchestrator-lambda-role --policy-arn arn:aws:iam::aws:policy/AmazonRDSFullAccess
```

> **Note:** These managed policies are used here for bootcamp simplicity. In a real production engagement, you'd write custom policies scoped to the specific resources (a single OpenSearch domain, a single RDS instance) rather than the full-access managed policies — the same least-privilege principle from Day 1, applied more strictly than time allows for in this lesson.

## Part 4: Deploying the Function

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

> **✅ Check:** Confirm your response includes `answer`, `route`, `sql_validation`, and `token_usage` fields, and that CloudWatch Logs show the decision trace from this afternoon's earlier lesson. This is the same deployment your capstone will build on directly.

## Resources

- [AWS Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

---

Conversion notes:
- `$VAR = (cmd)` → `VAR=$(cmd)` (no spaces around `=`).
- `Set-Content -Path f -Value '...'` → `cat > f <<'EOF' ... EOF` (quoted `EOF` prevents shell expansion of the JSON).
- `Get-Content f` → `cat f`.
- Backtick line-continuation → trailing backslash `\`.
- `arn:aws:iam::$ACCOUNT_ID`:role...` (PowerShell backtick-escaped colon) → plain `arn:aws:iam::$ACCOUNT_ID:role...`.