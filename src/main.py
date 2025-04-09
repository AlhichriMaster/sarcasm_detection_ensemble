import json
from data.sarcasm_data_generator import SarcasmDataGenerator
from models.ensemble_model import SarcasmEnsemble
from models.sarcasm_models import SarcasmModels
import numpy as np
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf

# Create results directory if it doesn't exist
results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Load the JSON dataset
def load_sarcasm_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item = json.loads(line)
            texts.append(item['headline'])
            labels.append(1 if item['is_sarcastic'] else 0)
    return np.array(texts), np.array(labels)

# Load and prepare the data
texts, labels = load_sarcasm_dataset('./Sarcasm_Headlines_Dataset.json')

# Split data into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, labels, test_size=0.3, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

# Initialize data generator
data_generator = SarcasmDataGenerator()
data_generator.fit_tokenizer(train_texts)

# Create data generators for training
BATCH_SIZE = 32
train_dataset = data_generator.generate_batches(train_texts, train_labels, batch_size=BATCH_SIZE)
val_dataset = data_generator.generate_batches(val_texts, val_labels, batch_size=BATCH_SIZE)
test_dataset = data_generator.generate_batches(test_texts, test_labels, batch_size=BATCH_SIZE, shuffle=False)

# Initialize models
model_builder = SarcasmModels(
    vocab_size=data_generator.max_words,
    max_len=data_generator.max_len
)

# Build individual models
models = {
    'LSTM': model_builder.build_lstm_model(),
    'Attention': model_builder.build_attention_model(),
    'Transformer': model_builder.build_transformer_model(1)
}

# Calculate steps per epoch
steps_per_epoch = len(train_texts) // BATCH_SIZE
validation_steps = len(val_texts) // BATCH_SIZE
test_steps = len(test_texts) // BATCH_SIZE

# Train and evaluate individual models
print("\n" + "="*50)
print("INDIVIDUAL MODEL RESULTS")
print("="*50)

# Create directory for saved models if it doesn't exist
saved_models_dir = 'saved_models'
if not os.path.exists(saved_models_dir):
    os.makedirs(saved_models_dir)

for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=5,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )
    
    # Save model weights
    model.save_weights(f'{saved_models_dir}/{model_name.lower()}.weights.h5')
    
    # Evaluate on test set
    test_results = model.evaluate(test_dataset, steps=test_steps)
    
    # Print results
    print(f"\n{model_name} Model Results:")
    print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")
    print("-" * 50)

# Create and train ensemble
print("\n" + "="*50)
print("ENSEMBLE MODEL RESULTS")
print("="*50)

ensemble = SarcasmEnsemble(list(models.values()))
ensemble_model = ensemble.build_ensemble()
ensemble_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train ensemble model
history = ensemble_model.fit(
    train_dataset, 
    validation_data=val_dataset,
    epochs=5,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)

# Evaluate ensemble model on test set
test_results = ensemble_model.evaluate(test_dataset, steps=test_steps)

# Print ensemble results
print("\nEnsemble Model Results:")
print(f"Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Test Accuracy: {test_results[1]:.4f}")
print("-" * 50)

print("\nAll models have been trained and evaluated.")