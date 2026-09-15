#!/usr/bin/env bash
#
# teardown.sh -- deletes every AWS resource this lesson creates.
#
# RDS and OpenSearch both bill continuously while running (unlike Lambda,
# which only bills for actual invocations) -- run this as soon as you're
# done testing, not at the end of the day.
#
# SAFETY: defaults to a dry run. Pass --confirm to actually delete anything.
#
# USAGE:
#   ./teardown.sh --name amw              # dry run, lists what would be deleted
#   ./teardown.sh --name amw --confirm    # actually deletes
#
# Replace "amw" with whatever <yourname> suffix you used when creating
# these resources in Parts 3-5 of the lesson.

set -euo pipefail

CONFIRM=false
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm) CONFIRM=true; shift ;;
    --name) NAME="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$NAME" ]]; then
  echo "Usage: $0 --name <yourname> [--confirm]"
  echo "Example: $0 --name amw --confirm"
  exit 1
fi

FUNCTION_NAME="multi-agent-orchestrator"
DB_IDENTIFIER="orchestrator-db-${NAME}"
DOMAIN_NAME="orchestrator-search-${NAME}"
ROLE_NAME="orchestrator-lambda-role-${NAME}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
BUCKET="lambda-deploy-${ACCOUNT_ID}-${NAME}"

echo "======================================================================"
if [[ "$CONFIRM" == false ]]; then
  echo "DRY RUN -- nothing will be deleted."
  echo "Review the list below, then re-run with --confirm to actually delete."
else
  echo "LIVE MODE -- resources WILL be deleted."
  read -p "Type DELETE (all caps) to confirm: " response
  if [[ "$response" != "DELETE" ]]; then
    echo "Confirmation not received. Exiting without changes."
    exit 0
  fi
fi
echo "======================================================================"

# --- Lambda function ---
echo ""
echo "--- Lambda function: ${FUNCTION_NAME} ---"
if aws lambda get-function --function-name "$FUNCTION_NAME" >/dev/null 2>&1; then
  echo "  Found."
  if [[ "$CONFIRM" == true ]]; then
    aws lambda delete-function --function-name "$FUNCTION_NAME"
    echo "  -> Deleted."
  fi
else
  echo "  Not found (already deleted, or different name used)."
fi

# --- RDS instance ---
echo ""
echo "--- RDS instance: ${DB_IDENTIFIER} ---"
if aws rds describe-db-instances --db-instance-identifier "$DB_IDENTIFIER" >/dev/null 2>&1; then
  STATUS=$(aws rds describe-db-instances --db-instance-identifier "$DB_IDENTIFIER" \
    --query "DBInstances[0].DBInstanceStatus" --output text)
  echo "  Found (status: ${STATUS}). Bills continuously while it exists."
  if [[ "$CONFIRM" == true ]]; then
    aws rds delete-db-instance --db-instance-identifier "$DB_IDENTIFIER" --skip-final-snapshot
    echo "  -> Deletion initiated (takes several minutes to fully complete)."
  fi
else
  echo "  Not found (already deleted, or different name used)."
fi

# --- OpenSearch domain ---
echo ""
echo "--- OpenSearch domain: ${DOMAIN_NAME} ---"
if aws opensearch describe-domain --domain-name "$DOMAIN_NAME" >/dev/null 2>&1; then
  echo "  Found. Bills continuously while it exists -- typically the most"
  echo "  expensive resource this lesson creates."
  if [[ "$CONFIRM" == true ]]; then
    aws opensearch delete-domain --domain-name "$DOMAIN_NAME"
    echo "  -> Deletion initiated (takes several minutes to fully complete)."
  fi
else
  echo "  Not found (already deleted, or different name used)."
fi

# --- S3 bucket (deployment package storage) ---
echo ""
echo "--- S3 bucket: ${BUCKET} ---"
if aws s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
  echo "  Found."
  if [[ "$CONFIRM" == true ]]; then
    aws s3 rm "s3://${BUCKET}" --recursive
    aws s3 rb "s3://${BUCKET}"
    echo "  -> Emptied and deleted."
  fi
else
  echo "  Not found (already deleted, or different name used)."
fi

# --- IAM role: informational only, cannot be deleted here ---
echo ""
echo "--- IAM role: ${ROLE_NAME} (informational only) ---"
if aws iam get-role --role-name "$ROLE_NAME" >/dev/null 2>&1; then
  echo "  Found. NOT deleted by this script -- iam:DeleteRolePolicy is blocked"
  echo "  in this sandbox, and a role can't be deleted while a policy is"
  echo "  attached to it. This role and its inline policy are permanent for"
  echo "  the life of this AWS account. This is not a cost concern: IAM"
  echo "  roles and policies themselves are free regardless of how many exist."
else
  echo "  Not found."
fi

echo ""
echo "======================================================================"
if [[ "$CONFIRM" == false ]]; then
  echo "Dry run complete. Nothing was deleted."
  echo "Re-run with --confirm when ready."
else
  echo "Teardown complete."
  echo "RDS and OpenSearch deletions take several minutes to finish in the"
  echo "background -- check the AWS console to confirm they've fully deleted"
  echo "before considering this done. The IAM role listed above remains and"
  echo "cannot be removed; this does not cost anything ongoing."
fi
echo "======================================================================"
