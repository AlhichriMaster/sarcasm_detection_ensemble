# ═══════════════════════════════════════════════════════════════════════════════
# Sarcasm Detection – Serverless AWS Deployment
# ═══════════════════════════════════════════════════════════════════════════════
#
# Prerequisites (Manjaro Linux):
#   sudo pacman -S docker aws-cli-v2
#   sudo systemctl start docker          # start Docker daemon
#   newgrp docker                        # add current user to docker group
#
# Usage:
#   1. Fill in aws_access_key / aws_secret_key in the locals block below.
#   2. cd terraform && terraform init && terraform apply
#
# What Terraform builds end-to-end:
#   ┌──────────────────────────────────────────────────────────────┐
#   │  local machine                                               │
#   │   • Fits tokenizer on dataset  → uploads to S3              │
#   │   • Uploads model weight files to S3                         │
#   │   • Builds Docker image (TF-CPU + src/)  → pushes to ECR    │
#   └──────────────────────────────────────────────────────────────┘
#                          │
#   AWS:  ECR → Lambda (container)  ←→  S3 (weights/tokenizer)
#                          │              DynamoDB (prediction log)
#         API Gateway POST /predict ──────────────────────────────►
#         CloudWatch alarm + SNS topic
# ═══════════════════════════════════════════════════════════════════════════════

terraform {
  required_version = ">= 1.3"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
    null = {
      source  = "hashicorp/null"
      version = "~> 3.0"
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# CREDENTIALS  ← update these two values before running terraform apply
# ─────────────────────────────────────────────────────────────────────────────
locals {
  aws_access_key = "" # e.g. AKIAIOSFODNN7EXAMPLE
  aws_secret_key = "" # e.g. wJalrXUtnFEMI/K7MDENG/...
  aws_region     = "us-east-1"

  # path.module = .../sarcasm_detection_ensemble/terraform/
  # project_root = .../sarcasm_detection_ensemble/
  project_root = abspath("${path.module}/..")

  # Stable resource names (match the tutorial)
  function_name = "SarcasmDetectionAPI"
  table_name    = "SarcasmPredictions"
  ecr_repo_name = "sarcasm-detection"
  image_tag     = "latest"
}

provider "aws" {
  region     = local.aws_region
  access_key = local.aws_access_key
  secret_key = local.aws_secret_key
}

# Convenience: every null_resource's AWS CLI calls use the same credentials
# as the Terraform provider, avoiding any mismatch with ~/.aws/credentials.
locals {
  cli_env = {
    AWS_ACCESS_KEY_ID     = local.aws_access_key
    AWS_SECRET_ACCESS_KEY = local.aws_secret_key
    AWS_DEFAULT_REGION    = local.aws_region
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SUFFIX for unique S3 bucket name
# ═══════════════════════════════════════════════════════════════════════════════
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ECR REPOSITORY
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_ecr_repository" "lambda_repo" {
  name                 = local.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  S3 BUCKET  (model weights + tokenizer)
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_s3_bucket" "model_bucket" {
  bucket        = "sarcasm-ml-models-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ── Upload pre-trained weight files (already on disk) ────────────────────────
resource "aws_s3_object" "lstm_weights" {
  bucket = aws_s3_bucket.model_bucket.id
  key    = "models/lstm.weights.h5"
  source = "${local.project_root}/saved_models/lstm.weights.h5"
  etag   = filemd5("${local.project_root}/saved_models/lstm.weights.h5")
}

resource "aws_s3_object" "attention_weights" {
  bucket = aws_s3_bucket.model_bucket.id
  key    = "models/attention.weights.h5"
  source = "${local.project_root}/saved_models/attention.weights.h5"
  etag   = filemd5("${local.project_root}/saved_models/attention.weights.h5")
}

resource "aws_s3_object" "transformer_weights" {
  bucket = aws_s3_bucket.model_bucket.id
  key    = "models/transformer.weights.h5"
  source = "${local.project_root}/saved_models/transformer.weights.h5"
  etag   = filemd5("${local.project_root}/saved_models/transformer.weights.h5")
}

# ── Generate tokenizer.pkl locally then upload to S3 ─────────────────────────
# Fits a Keras Tokenizer on all 26 k headlines and serialises it with pickle.
# The Lambda downloads this at cold-start so it doesn't need the full dataset.
# BUCKET_NAME is injected via the environment block so the command is portable.
resource "null_resource" "tokenizer" {
  triggers = {
    dataset_hash = filemd5("${local.project_root}/Sarcasm_Headlines_Dataset.json")
    script_hash  = filemd5("${path.module}/gen_tokenizer.py")
    bucket_id    = aws_s3_bucket.model_bucket.id
  }

  provisioner "local-exec" {
    # Run from project root so gen_tokenizer.py can open relative paths
    working_dir = local.project_root
    environment = local.cli_env
    command     = <<-BASH
      set -e
      echo "==> Installing tokenizer dependencies locally..."
      pip3 install --quiet --user --break-system-packages "numpy>=1.24,<2.0" "keras-preprocessing" 2>&1 | tail -3
      echo "==> Generating tokenizer..."
      python3 "${abspath(path.module)}/gen_tokenizer.py"
      echo "==> Uploading tokenizer to s3://${aws_s3_bucket.model_bucket.id}/models/tokenizer.pkl"
      aws s3 cp exports/tokenizer.pkl "s3://${aws_s3_bucket.model_bucket.id}/models/tokenizer.pkl"
      echo "==> Tokenizer upload complete."
    BASH
  }

  depends_on = [aws_s3_bucket.model_bucket]
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  BUILD & PUSH DOCKER IMAGE TO ECR
# ═══════════════════════════════════════════════════════════════════════════════
# The Docker build context is the PROJECT ROOT so that
#   COPY src/  ...
#   COPY terraform/lambda_function.py  ...
# both resolve correctly.  The Dockerfile lives in terraform/.
resource "null_resource" "build_and_push_image" {
  triggers = {
    # Rebuild whenever handler, Dockerfile, or any source file changes
    handler_hash    = filemd5("${path.module}/lambda_function.py")
    dockerfile_hash = filemd5("${path.module}/Dockerfile")
    src_hash = sha256(join("", [
      filemd5("${local.project_root}/src/models/sarcasm_models.py"),
      filemd5("${local.project_root}/src/models/ensemble_model.py"),
      filemd5("${local.project_root}/src/data/sarcasm_data_generator.py"),
    ]))
  }

  provisioner "local-exec" {
    working_dir = local.project_root
    environment = merge(local.cli_env, {
      REPO_URL = aws_ecr_repository.lambda_repo.repository_url
    })
    interpreter = ["/bin/bash", "-c"]
    command     = <<-BASH
      set -e

      echo "==> Authenticating with ECR..."
      aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | \
        docker login --username AWS --password-stdin "$REPO_URL"

      echo "==> Building Docker image (linux/amd64)..."
      docker build \
        --no-cache \
        --platform linux/amd64 \
        -f terraform/Dockerfile \
        -t sarcasm-detection:latest \
        .

      echo "==> Tagging image..."
      docker tag sarcasm-detection:latest "$REPO_URL:latest"

      echo "==> Pushing to ECR..."
      docker push "$REPO_URL:latest"
      echo "==> Image pushed successfully."
    BASH
  }

  depends_on = [
    aws_ecr_repository.lambda_repo,
  ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  DYNAMODB TABLE  (prediction log)
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_dynamodb_table" "predictions" {
  name         = local.table_name
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "prediction_id"
  range_key    = "timestamp"

  attribute {
    name = "prediction_id"
    type = "S"
  }
  attribute {
    name = "timestamp"
    type = "S"
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  IAM ROLE & POLICY  (Lambda execution)
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_iam_role" "lambda_role" {
  name = "SarcasmDetectionLambdaRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "SarcasmDetectionPermissions"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3ModelAccess"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.model_bucket.arn,
          "${aws_s3_bucket.model_bucket.arn}/*",
        ]
      },
      {
        Sid      = "DynamoDBWrite"
        Effect   = "Allow"
        Action   = ["dynamodb:PutItem", "dynamodb:GetItem", "dynamodb:Query"]
        Resource = aws_dynamodb_table.predictions.arn
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        # Lambda needs these to pull the container image from ECR at startup
        Sid    = "ECRPull"
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
        ]
        Resource = aws_ecr_repository.lambda_repo.arn
      },
      {
        Sid      = "ECRAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
    ]
  })
}

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  CLOUDWATCH LOG GROUP
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_cloudwatch_log_group" "lambda_logs" {
  name              = "/aws/lambda/${local.function_name}"
  retention_in_days = 14
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  LAMBDA FUNCTION  (container image)
# ═══════════════════════════════════════════════════════════════════════════════
# Cold-start note:
#   First invocation downloads weights from S3 and builds three Keras models
#   (~60-120 s).  The warm-up step at the end pre-loads the model via a direct
#   Lambda invocation (no 29-second API Gateway limit applies there).
#   Subsequent API calls are fast (<1 s) on warm instances.
resource "aws_lambda_function" "sarcasm_api" {
  function_name = local.function_name
  role          = aws_iam_role.lambda_role.arn
  package_type  = "Image"

  # Reference the image pushed by null_resource.build_and_push_image.
  # On re-applies, Terraform will detect if the image_uri attribute changes
  # (e.g. after a re-tag); use `terraform taint null_resource.build_and_push_image`
  # to force a rebuild + re-deploy when source code changes.
  image_uri = "${aws_ecr_repository.lambda_repo.repository_url}:${local.image_tag}"

  timeout     = 300  # 5 min – covers the cold-start model build
  memory_size = 3008 # Maximum Lambda memory for fastest model loading

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.model_bucket.id
      TABLE_NAME   = local.table_name
    }
  }

  depends_on = [
    aws_iam_role_policy.lambda_policy,
    aws_cloudwatch_log_group.lambda_logs,
    null_resource.build_and_push_image,
    null_resource.tokenizer,
    aws_s3_object.lstm_weights,
    aws_s3_object.attention_weights,
    aws_s3_object.transformer_weights,
  ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  API GATEWAY  (REST API – POST /predict)
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_api_gateway_rest_api" "api" {
  name        = "Sarcasm Detection API"
  description = "Ensemble neural network sarcasm classifier (LSTM + Attention + Transformer)"
}

resource "aws_api_gateway_resource" "predict" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "predict"
}

# ── POST /predict  (Lambda proxy integration) ─────────────────────────────────
resource "aws_api_gateway_method" "post" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id             = aws_api_gateway_rest_api.api.id
  resource_id             = aws_api_gateway_resource.predict.id
  http_method             = aws_api_gateway_method.post.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.sarcasm_api.invoke_arn
  timeout_milliseconds    = 29000 # API Gateway hard maximum
}

# ── OPTIONS /predict  (CORS preflight) ────────────────────────────────────────
resource "aws_api_gateway_method" "options" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.predict.id
  http_method   = "OPTIONS"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "options_mock" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.options.http_method
  type        = "MOCK"

  request_templates = {
    "application/json" = "{\"statusCode\": 200}"
  }
}

resource "aws_api_gateway_method_response" "options_200" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = "200"

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = true
    "method.response.header.Access-Control-Allow-Methods" = true
    "method.response.header.Access-Control-Allow-Origin"  = true
  }
}

resource "aws_api_gateway_integration_response" "options_response" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.predict.id
  http_method = aws_api_gateway_method.options.http_method
  status_code = aws_api_gateway_method_response.options_200.status_code

  response_parameters = {
    "method.response.header.Access-Control-Allow-Headers" = "'Content-Type,X-Amz-Date,Authorization'"
    "method.response.header.Access-Control-Allow-Methods" = "'GET,OPTIONS,POST'"
    "method.response.header.Access-Control-Allow-Origin"  = "'*'"
  }

  depends_on = [aws_api_gateway_integration.options_mock]
}

# ── Deployment → prod stage ───────────────────────────────────────────────────
resource "aws_api_gateway_deployment" "deployment" {
  rest_api_id = aws_api_gateway_rest_api.api.id

  # Force a new deployment whenever any API resource changes
  triggers = {
    redeployment = sha1(jsonencode([
      aws_api_gateway_resource.predict.id,
      aws_api_gateway_method.post.id,
      aws_api_gateway_integration.lambda_integration.id,
      aws_api_gateway_integration.options_mock.id,
      aws_api_gateway_integration_response.options_response.id,
    ]))
  }

  lifecycle {
    create_before_destroy = true
  }

  depends_on = [
    aws_api_gateway_method.post,
    aws_api_gateway_integration.lambda_integration,
    aws_api_gateway_integration_response.options_response,
  ]
}

resource "aws_api_gateway_stage" "prod" {
  deployment_id = aws_api_gateway_deployment.deployment.id
  rest_api_id   = aws_api_gateway_rest_api.api.id
  stage_name    = "prod"
}

# Grant API Gateway permission to invoke the Lambda
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sarcasm_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.api.execution_arn}/*/*"
}

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  MONITORING  (CloudWatch alarm + SNS topic)
# ═══════════════════════════════════════════════════════════════════════════════
resource "aws_sns_topic" "alerts" {
  name = "SarcasmAlerts"
}

resource "aws_cloudwatch_metric_alarm" "high_invocations" {
  alarm_name          = "SarcasmAPI-HighInvocations"
  alarm_description   = "Lambda invocations exceeded 1 000 per hour"
  namespace           = "AWS/Lambda"
  metric_name         = "Invocations"
  statistic           = "Sum"
  period              = 3600
  evaluation_periods  = 1
  threshold           = 1000
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    FunctionName = aws_lambda_function.sarcasm_api.function_name
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  WARM-UP INVOCATION
# ═══════════════════════════════════════════════════════════════════════════════
# Directly invokes Lambda (not through API Gateway), so the 300-second timeout
# applies instead of API Gateway's 29-second cap.  After this completes, the
# model is cached and all subsequent API calls are fast.
resource "null_resource" "warm_up_lambda" {
  triggers = {
    lambda_arn = aws_lambda_function.sarcasm_api.arn
  }

  provisioner "local-exec" {
    environment = merge(local.cli_env, {
      FUNCTION_NAME = local.function_name
    })
    interpreter = ["/bin/bash", "-c"]
    command     = <<-BASH
      echo "==> Warming up Lambda (loading TF models – may take 2-3 min)..."
      aws lambda invoke \
        --function-name "$FUNCTION_NAME" \
        --payload '{"text":"Scientists shocked to discover water is still wet"}' \
        /tmp/sarcasm_warmup.json
      echo "==> Warm-up response:"
      cat /tmp/sarcasm_warmup.json
      echo ""
      echo "==> Lambda is warm. API Gateway calls will now respond quickly."
    BASH
  }

  depends_on = [
    aws_lambda_function.sarcasm_api,
    aws_lambda_permission.api_gateway,
  ]
}

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUTS
# ═══════════════════════════════════════════════════════════════════════════════
output "api_endpoint" {
  description = "REST API endpoint – POST JSON {\"text\": \"headline\"} here"
  value       = "https://${aws_api_gateway_rest_api.api.id}.execute-api.${local.aws_region}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}/predict"
}

output "curl_example" {
  description = "Smoke-test command (run after terraform apply)"
  value       = <<-EOT
    curl -s -X POST \
      "https://${aws_api_gateway_rest_api.api.id}.execute-api.${local.aws_region}.amazonaws.com/${aws_api_gateway_stage.prod.stage_name}/predict" \
      -H "Content-Type: application/json" \
      -d '{"text": "Man who thought losing was impossible loses everything"}' \
      | python3 -m json.tool
  EOT
}

output "s3_bucket" {
  description = "S3 bucket storing model weights and tokenizer"
  value       = aws_s3_bucket.model_bucket.id
}

output "dynamodb_table" {
  description = "DynamoDB table for prediction logs"
  value       = aws_dynamodb_table.predictions.name
}

output "ecr_repository" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.lambda_repo.repository_url
}

output "lambda_function" {
  description = "Lambda function name"
  value       = aws_lambda_function.sarcasm_api.function_name
}

output "sns_topic_arn" {
  description = "Subscribe an email here to receive high-traffic alerts"
  value       = aws_sns_topic.alerts.arn
}
