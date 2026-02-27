"""
Fit a Keras Tokenizer on the sarcasm headlines dataset and save it as
exports/tokenizer.pkl so the Lambda function can load it from S3.
Run this script from the project root directory:
    python3 terraform/gen_tokenizer.py

Requires only: pip install numpy keras-preprocessing
(keras_preprocessing.text.Tokenizer is the same class that TensorFlow
exposes as tensorflow.keras.preprocessing.text.Tokenizer, so the
pickled object is fully compatible with the Lambda runtime.)
"""
import json
import os
import pickle
import re

from keras_preprocessing.text import Tokenizer

MAX_WORDS = 10000
MAX_LEN   = 100

project_root = os.getcwd()
dataset_path = os.path.join(project_root, "Sarcasm_Headlines_Dataset.json")

texts = []
with open(dataset_path, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(json.loads(line)["headline"])

# Mirror the preprocessing done by SarcasmDataGenerator.preprocess_text
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

processed = [preprocess(t) for t in texts]

print(f"Fitting tokenizer on {len(processed)} headlines...")
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(processed)

export_dir = os.path.join(project_root, "exports")
os.makedirs(export_dir, exist_ok=True)
out_path = os.path.join(export_dir, "tokenizer.pkl")
with open(out_path, "wb") as f:
    pickle.dump({"tokenizer": tokenizer, "max_len": MAX_LEN}, f)

print(f"tokenizer.pkl written to {out_path}")
