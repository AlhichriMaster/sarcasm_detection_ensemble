"""
Sarcasm Detection Lambda Handler
---------------------------------
Cold start: downloads model weights from S3, builds the three-model ensemble
(LSTM + Attention + Transformer) in memory, caches globally in the Lambda
execution environment for fast subsequent (warm) invocations.
Predictions are logged to DynamoDB.
"""
import json
import os
import pickle
import re
import sys
import uuid
import boto3
from datetime import datetime

import numpy as np

# Make the project source code importable inside the Lambda container.
# The Dockerfile copies src/ into ${LAMBDA_TASK_ROOT}/src/.
sys.path.insert(0, "/var/task/src")

s3_client = boto3.client("s3")
dynamodb  = boto3.resource("dynamodb")

MODEL_BUCKET = os.environ["MODEL_BUCKET"]
TABLE_NAME   = os.environ.get("TABLE_NAME", "SarcasmPredictions")

# ── Cached globals (persist across warm invocations) ─────────────────────────
_ensemble_model   = None
_tokenizer_config = None


def _download_if_missing(bucket: str, s3_key: str, local_path: str) -> None:
    if not os.path.exists(local_path):
        print(f"Downloading s3://{bucket}/{s3_key} → {local_path}")
        s3_client.download_file(bucket, s3_key, local_path)


def _load_models() -> None:
    """Download weights + tokenizer from S3, build ensemble, cache globally."""
    global _ensemble_model, _tokenizer_config

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if _tokenizer_config is None:
        _download_if_missing(MODEL_BUCKET, "models/tokenizer.pkl", "/tmp/tokenizer.pkl")
        with open("/tmp/tokenizer.pkl", "rb") as f:
            _tokenizer_config = pickle.load(f)
        print("Tokenizer loaded.")

    # ── Model weights → ensemble ──────────────────────────────────────────────
    if _ensemble_model is None:
        import tensorflow as tf
        from models.sarcasm_models import SarcasmModels
        from models.ensemble_model import SarcasmEnsemble

        weight_keys = {
            "lstm":        "models/lstm.weights.h5",
            "attention":   "models/attention.weights.h5",
            "transformer": "models/transformer.weights.h5",
        }
        for name, s3_key in weight_keys.items():
            _download_if_missing(MODEL_BUCKET, s3_key, f"/tmp/{name}.weights.h5")

        builder = SarcasmModels(vocab_size=10000, max_len=100)
        models = {
            "lstm":        builder.build_lstm_model(),
            "attention":   builder.build_attention_model(),
            "transformer": builder.build_transformer_model(num_blocks=1),
        }

        # A forward pass through a dummy input is required before load_weights()
        # so that Keras builds the model's weight tensors.
        dummy = tf.zeros((1, 100), dtype=tf.int32)
        for name, model in models.items():
            model(dummy)
            model.load_weights(f"/tmp/{name}.weights.h5")
            print(f"{name} weights loaded.")

        ens = SarcasmEnsemble(list(models.values()))
        _ensemble_model = ens.build_ensemble()
        _ensemble_model(dummy)  # finalise graph
        print("Ensemble model ready.")


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _predict(text: str) -> dict:
    processed = _preprocess(text)
    tokenizer = _tokenizer_config["tokenizer"]
    max_len   = _tokenizer_config["max_len"]

    seq    = tokenizer.texts_to_sequences([processed])
    padded = np.zeros((1, max_len), dtype=np.int32)
    if seq[0]:
        seq_len = min(len(seq[0]), max_len)
        padded[0, :seq_len] = seq[0][:seq_len]

    prob = float(_ensemble_model.predict(padded, verbose=0)[0][0])
    return {
        "probability":  round(prob, 4),
        "is_sarcastic": bool(prob > 0.5),
        "confidence":   round(abs(prob - 0.5) * 2, 4),
    }


def _log_prediction(text: str, prediction: dict) -> None:
    try:
        dynamodb.Table(TABLE_NAME).put_item(Item={
            "prediction_id": str(uuid.uuid4()),
            "timestamp":     datetime.utcnow().isoformat(),
            "input_text":    text[:1000],
            "probability":   str(prediction["probability"]),
            "is_sarcastic":  prediction["is_sarcastic"],
            "confidence":    str(prediction["confidence"]),
        })
    except Exception as exc:
        print(f"DynamoDB logging error: {exc}")


def lambda_handler(event, context):
    try:
        # API Gateway wraps the payload in a JSON-string 'body' field.
        # Direct Lambda invocations use the root object as the payload.
        body = event
        if "body" in event:
            raw  = event["body"]
            body = json.loads(raw) if isinstance(raw, str) else raw

        text = (body.get("text") or "").strip()
        if not text:
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {"error": 'Missing "text" field. Send JSON: {"text": "your headline"}'}
                ),
            }

        _load_models()
        prediction = _predict(text)
        _log_prediction(text, prediction)

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"input_text": text, "prediction": prediction}),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(exc)}),
        }
