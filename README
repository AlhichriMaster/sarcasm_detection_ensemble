# Sarcasm Detection using Neural Network Ensemble

This repository contains a TensorFlow implementation of an ensemble neural network for detecting sarcasm in news headlines. The project implements and compares three different neural architectures (LSTM, Attention, and Transformer) and combines them into an ensemble model for improved performance.

The model is deployed as a **fully serverless REST API on AWS**, allowing anyone to submit a news headline and receive a sarcasm prediction in seconds — no infrastructure to manage.

---

## 🎬 Demo

Watch a live walkthrough of the API in action: **https://youtu.be/6QDDyuPzelE**

---

## Project Structure

```
sarcasm-detection/
├── src/
│   ├── data/
│   │   └── sarcasm_data_generator.py
│   └── models/
│       ├── ensemble_model.py
│       └── sarcasm_models.py
├── saved_models/
│   ├── lstm.weights.h5
│   ├── attention.weights.h5
│   └── transformer.weights.h5
├── terraform/
│   ├── main.tf
│   └── lambda_function.py
├── results/
├── exports/
├── Sarcasm_Headlines_Dataset.json
└── README.md
```

---

## Cloud Architecture

The sarcasm detection model is deployed as a serverless application on AWS. A user submits a news headline to a REST API and within seconds receives a prediction.

![Cloud Architecture Diagram](Sarcasm_diagram.png)

### Services at a Glance

| Service | Role |
|---|---|
| **Amazon API Gateway** | Exposes the REST `/predict` endpoint to clients |
| **AWS Lambda** | Runs the ensemble inference logic (container image) |
| **Amazon ECR** | Hosts the Docker image containing TensorFlow and the model code |
| **Amazon S3** | Stores model weights and the fitted tokenizer |
| **Amazon DynamoDB** | Logs every prediction (text, probability, timestamp) |
| **Amazon CloudWatch** | Monitors Lambda errors and latency with alarms |
| **Amazon SNS** | Sends alert notifications when CloudWatch alarms trigger |

### Request Flow

1. A client sends a `POST /predict` request with a JSON body to **API Gateway**
2. API Gateway triggers an **AWS Lambda** function
3. On cold start, Lambda pulls the Docker image from **ECR** and downloads model weights from **S3**
4. Lambda runs the three-model ensemble and returns a prediction JSON
5. The prediction record is written to **DynamoDB**
6. **CloudWatch** monitors Lambda health; **SNS** fires alerts on anomalies

---

## Deployment

The entire system is provisioned with Terraform — no manual console clicks required. A single `terraform apply` builds the Docker image, pushes it to ECR, uploads model files to S3, creates the DynamoDB table, wires up API Gateway, and runs a warm-up invocation so the first real request is fast.

### Prerequisites

```bash
sudo pacman -S docker aws-cli-v2   # or brew install / apt install equivalents
sudo systemctl start docker
newgrp docker   # or log out/in
```

### Deploy

```bash
# 1. Fill in your AWS credentials
nano terraform/main.tf   # set aws_access_key and aws_secret_key

# 2. Deploy (takes ~10–15 min for the first Docker build)
cd terraform
terraform init
terraform apply
```

After apply completes, Terraform prints the live API URL.

### Tear Down

```bash
terraform destroy
```

This removes all AWS resources and stops all charges.

---

## Using the API

The API accepts `POST` requests to the `/predict` endpoint.

**Endpoint:**
```
POST https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/predict
```

**Request Body:**
```json
{ "text": "Your headline here" }
```

**Example with curl:**
```bash
curl -s -X POST \
  "https://<api-id>.execute-api.us-east-1.amazonaws.com/prod/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Man Who Thought Losing Was Impossible Loses Everything"}'
```

**Example Response:**
```json
{
  "text":         "Man Who Thought Losing Was Impossible Loses Everything",
  "is_sarcastic": true,
  "probability":  0.87,
  "confidence":   74.0
}
```

---

## Design Decisions

**Why AWS?** AWS offers tightly integrated serverless services (Lambda, API Gateway, S3, DynamoDB, ECR, CloudWatch, SNS) with the most mature serverless ML deployment ecosystem and richest Terraform provider support.

**Why Lambda over EC2/ECS?** Lambda scales automatically from zero to thousands of concurrent requests and costs approximately $1.20/month at 10,000 requests, compared to ~$30/month for a 24/7 EC2 instance — roughly 25x cheaper.

**Why a Container Image Lambda?** Standard Lambda packages have a 250 MB limit; TensorFlow alone exceeds this. Container-image Lambdas support up to 10 GB, comfortably fitting TensorFlow-CPU 2.15, NumPy, Boto3, and the model code.

**Why S3 for Model Storage?** Model weights are large binary artifacts that change independently from application code. Lambda downloads them to its `/tmp` directory on cold start and keeps them in memory for warm requests, so S3 is only hit once per container lifecycle.

**Why DynamoDB for Prediction Logs?** The prediction log is a simple append-only record with no complex queries. DynamoDB requires zero administration, has no minimum cost when idle, and scales automatically — unlike RDS which would add $15–30/month minimum.

**Why Terraform?** Manual console clicks produce infrastructure that is hard to reproduce or roll back. Terraform describes the entire system in version-controlled code; a new team member can run `terraform apply` and get an identical environment in under 15 minutes.

---

## Local Development

### Requirements

- Python 3.8+
- TensorFlow 2.4+
- NumPy
- scikit-learn

```bash
pip install tensorflow numpy scikit-learn
```

### Train Models

```bash
python main.py
```

This loads and preprocesses the dataset, trains the LSTM, Attention, and Transformer models, saves weights to `saved_models/`, evaluates the ensemble, and saves results to `results/`.

### Run the Demo (Local)

```bash
python demo.py
```

Loads pre-trained weights and prompts you to enter headlines, returning predictions with confidence scores from each model and the ensemble.

---

## Dataset

The project uses the "Sarcasm Detection through NLP" dataset (~28,000 labeled news headlines from TheOnion and HuffPost), included as `Sarcasm_Headlines_Dataset.json`.

---

## Performance

With optimized hyperparameters, the ensemble achieves approximately **85.8% accuracy** on the test set:

| Model | Accuracy |
|---|---|
| LSTM | 84.7% |
| Attention | 84.0% |
| Transformer | 82.4% |
| **Ensemble** | **85.8%** |

Each architecture captures sarcasm differently — the LSTM detects sequential patterns, the attention model links specific word pairs, and the transformer sees the full headline at once. Averaging their predictions reduces overall error by cancelling out each model's blind spots.

---

## Limitations

- The model was trained on news headlines only and may not generalize well to social media posts or product reviews.
- Headlines requiring real-world knowledge to identify as sarcastic may be misclassified.
- The API does not require authentication.
- Lambda cold starts still occur if the function has been idle for more than ~15 minutes.
